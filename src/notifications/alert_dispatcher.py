
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pika
from dotenv import dotenv_values
from elasticsearch import Elasticsearch
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from . import notify_gmail, notify_line

ROOT = Path(__file__).resolve().parent.parent.parent
NOTIFICATION_ENV_PATH = ROOT / "notification" / ".env"
ROOT_ENV_PATH = ROOT / ".env"
DIGEST_DB_PATH = ROOT / "data" / "notification" / "low_priority_digest.db"

VALID_PRIORITIES = ("HIGH", "MEDIUM", "LOW")

logger = logging.getLogger("alert_dispatcher")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
_log_level = (dotenv_values(ROOT_ENV_PATH).get("LOG_LEVEL") or "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))


REQUIRED_ENV_KEYS = (
    "RABBITMQ_URL",
    "RABBITMQ_VHOST",
    "RABBITMQ_EXCHANGE",
    "INFLUX_URL",
    "INFLUX_TOKEN",
    "INFLUX_ORG",
    "INFLUX_BUCKET",
    "ELASTIC_URL",
    "ELASTIC_API_KEY",
    "ELASTIC_ALERT_INDEX",
    "LINE_CHANNEL_ACCESS_TOKEN",
    "LINE_TO_ID",
    "LINE_GRAFANA_BASE_URL",
    "GMAIL_CLIENT_ID",
    "GMAIL_CLIENT_SECRET",
    "GMAIL_REFRESH_TOKEN",
    "GMAIL_SENDER_ADDRESS",
    "LOW_DIGEST_SCHEDULE",
)


def load_env_config(env_path: Path = NOTIFICATION_ENV_PATH) -> dict[str, Any]:
    raw = dotenv_values(env_path)
    missing = [k for k in REQUIRED_ENV_KEYS if k not in raw or raw[k] is None]
    if missing:
        raise RuntimeError(f"notification/.env is missing required keys: {missing}")

    return {
        "rabbitmq_url": raw["RABBITMQ_URL"],
        "rabbitmq_vhost": raw["RABBITMQ_VHOST"],
        "rabbitmq_exchange": raw["RABBITMQ_EXCHANGE"],
        "influx_url": raw["INFLUX_URL"],
        "influx_token": raw["INFLUX_TOKEN"],
        "influx_org": raw["INFLUX_ORG"],
        "influx_bucket": raw["INFLUX_BUCKET"],
        "elastic_url": raw["ELASTIC_URL"],
        "elastic_api_key": raw["ELASTIC_API_KEY"],
        "elastic_alert_index_pattern": raw["ELASTIC_ALERT_INDEX"],
        "line_channel_access_token": raw["LINE_CHANNEL_ACCESS_TOKEN"],
        "line_to_id": raw["LINE_TO_ID"],
        "line_grafana_base_url": raw["LINE_GRAFANA_BASE_URL"],
        "gmail_client_id": raw["GMAIL_CLIENT_ID"],
        "gmail_client_secret": raw["GMAIL_CLIENT_SECRET"],
        "gmail_refresh_token": raw["GMAIL_REFRESH_TOKEN"],
        "gmail_sender_address": raw["GMAIL_SENDER_ADDRESS"],
        "low_digest_schedule": raw["LOW_DIGEST_SCHEDULE"],
    }


def derive_model_used(alert_context: dict[str, Any], llm_output: dict[str, Any]) -> str:
    model_used = alert_context.get("model_used")
    if model_used not in ("random_forest", "autoencoder_v2"):
        logger.warning("[alert_dispatcher] no valid pipeline model_used -- model_used='unknown'")
        return "unknown"
    return model_used


def _elapsed_ms(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    return max(0.0, (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds() * 1000)


def assemble_alert_record(
    alert_context: dict[str, Any],
    fused_score: float,
    llm_output: dict[str, Any],
    model_used: str,
    alert_id: str,
) -> dict[str, Any]:
    display_score = min(max(float(fused_score), 0.0), 1.0)
    risk_score = round(display_score * 100)
    ingested_at = alert_context.get("ingested_at")
    detected_at = alert_context.get("detected_at")
    return {
        "alert_id": alert_id,
        "timestamp": alert_context.get("timestamp"),
        "priority": llm_output["priority"],
        "risk_score": risk_score,
        "fused_score": float(fused_score),
        "fused_prediction": alert_context.get("fused_prediction"),
        "source_ip": alert_context.get("source_ip"),
        "dst_ip": alert_context.get("dst_ip"),
        "dst_port": alert_context.get("dst_port"),
        "affected_asset": llm_output.get("affected_asset") or alert_context.get("affected_asset"),
        "anomaly_type": alert_context.get("rf_predicted_label"),
        "technique_id": llm_output.get("technique_id"),
        "mitre_tactic": llm_output.get("mitre_tactic"),
        "explanation_th": llm_output.get("explanation_th"),
        "remediation": llm_output.get("remediation") or [],
        "confidence": llm_output.get("confidence", alert_context.get("confidence")),
        "response_ms": llm_output.get("response_ms"),
        "ingested_at": ingested_at,
        "detected_at": detected_at,
        "mttd_ms": _elapsed_ms(ingested_at, detected_at),
        "model_used": model_used,
    }


def publish_to_rabbitmq(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    routing_key = f"alert.{record['priority']}"
    params = pika.URLParameters(config["rabbitmq_url"])
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        channel.exchange_declare(exchange=config["rabbitmq_exchange"], exchange_type="topic", durable=True)
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        channel.basic_publish(
            exchange=config["rabbitmq_exchange"],
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        )
    finally:
        connection.close()
    return {"status": "ok", "exchange": config["rabbitmq_exchange"], "vhost": config["rabbitmq_vhost"], "routing_key": routing_key}


def write_influx_point(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    ts = record.get("timestamp")
    point_time = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)

    point = (
        Point("soc_alert")
        .tag("alert_id", str(record.get("alert_id") or "unknown"))
        .tag("source_ip", str(record.get("source_ip") or "unknown"))
        .tag("dst_port", str(record.get("dst_port") if record.get("dst_port") is not None else "unknown"))
        .tag("priority", str(record.get("priority") or "unknown"))
        .tag("anomaly_type", str(record.get("anomaly_type") or "unknown"))
        .tag("mitre_tactic", str(record.get("mitre_tactic") or "unknown"))
        .field("risk_score", float(record.get("risk_score") or 0))
        .field("response_ms", float(record.get("response_ms") or 0.0))
        .time(point_time)
    )

    with InfluxDBClient(url=config["influx_url"], token=config["influx_token"], org=config["influx_org"]) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=config["influx_bucket"], record=point)

    return {"status": "ok", "measurement": "soc_alert", "bucket": config["influx_bucket"], "time": point_time.isoformat()}


def _es_index_name(timestamp_iso: str | None) -> str:
    dt = datetime.fromisoformat(timestamp_iso) if timestamp_iso else datetime.now(timezone.utc)
    return f"soc-alerts-{dt.strftime('%Y.%m.%d')}"


def write_elasticsearch_doc(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    index_name = _es_index_name(record.get("timestamp"))
    es = Elasticsearch(config["elastic_url"], api_key=config["elastic_api_key"])
    resp = es.index(index=index_name, id=record["alert_id"], document=record)
    return {"status": "ok", "index": index_name, "result": resp.get("result")}


def _get_digest_conn() -> sqlite3.Connection:
    DIGEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DIGEST_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS low_priority_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source_ip TEXT,
            dst_ip TEXT,
            dst_port TEXT,
            affected_asset TEXT,
            anomaly_type TEXT,
            mitre_tactic TEXT,
            technique_id TEXT,
            risk_score REAL,
            response_ms REAL,
            mttd_ms REAL,
            remediation_json TEXT,
            explanation_th TEXT,
            flushed_at TEXT,
            digest_batch_id TEXT
        )
        """
    )
    if "mttd_ms" not in {row[1] for row in conn.execute("PRAGMA table_info(low_priority_alerts)")}:
        conn.execute("ALTER TABLE low_priority_alerts ADD COLUMN mttd_ms REAL")
    conn.commit()
    return conn


def accumulate_low_priority_digest(record: dict[str, Any]) -> dict[str, Any]:
    conn = _get_digest_conn()
    try:
        conn.execute(
            """
            INSERT INTO low_priority_alerts
                (alert_id, timestamp, source_ip, dst_ip, dst_port, affected_asset, anomaly_type,
                 mitre_tactic, technique_id, risk_score, response_ms, mttd_ms, remediation_json, explanation_th)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("alert_id"),
                record.get("timestamp"),
                record.get("source_ip"),
                record.get("dst_ip"),
                str(record.get("dst_port")) if record.get("dst_port") is not None else None,
                record.get("affected_asset"),
                record.get("anomaly_type"),
                record.get("mitre_tactic"),
                record.get("technique_id"),
                record.get("risk_score"),
                record.get("response_ms"),
                record.get("mttd_ms"),
                json.dumps(record.get("remediation") or [], ensure_ascii=False),
                record.get("explanation_th"),
            ),
        )
        conn.commit()
        row_count = conn.execute("SELECT COUNT(*) FROM low_priority_alerts WHERE flushed_at IS NULL").fetchone()[0]
    finally:
        conn.close()
    return {"status": "accumulated", "db_path": str(DIGEST_DB_PATH), "pending_count": row_count}


def flush_low_priority_digest(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_env_config()
    conn = _get_digest_conn()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM low_priority_alerts WHERE flushed_at IS NULL ORDER BY timestamp").fetchall()
        if not rows:
            return {"status": "no_alerts", "count": 0}

        row_dicts = [dict(r) for r in rows]
        type_counts = Counter(r["anomaly_type"] or "unknown" for r in row_dicts)
        mttd_values = [r["mttd_ms"] for r in row_dicts if r["mttd_ms"] is not None]
        avg_mttd_ms = sum(mttd_values) / len(mttd_values) if mttd_values else None
        sample_remediation: list[str] = []
        for r in row_dicts:
            for step in json.loads(r["remediation_json"] or "[]"):
                if step not in sample_remediation:
                    sample_remediation.append(step)
            if len(sample_remediation) >= 5:
                break

        stats = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "alert_count": len(row_dicts),
            "top_attack_types": type_counts.most_common(5),
            "avg_mttd_ms": avg_mttd_ms,
            "sample_remediation_steps": sample_remediation[:5],
        }

        notif_result = notify_gmail.send_digest_email(row_dicts, stats, config)
        if notif_result.get("status") != "sent":
            return {
                "status": "dry_run" if notif_result.get("status") == "dry_run" else "error",
                "count": len(row_dicts), "stats": stats, "notification": notif_result,
            }

        batch_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "UPDATE low_priority_alerts SET flushed_at = ?, digest_batch_id = ? "
            "WHERE id = ? AND flushed_at IS NULL",
            [(now_iso, batch_id, r["id"]) for r in row_dicts],
        )
        conn.commit()

        return {"status": "flushed", "count": len(row_dicts), "digest_batch_id": batch_id, "stats": stats, "notification": notif_result}
    finally:
        conn.close()


def dispatch_alert(alert_context: dict[str, Any], fused_score: float, llm_output: dict[str, Any]) -> dict[str, Any]:
    priority = llm_output.get("priority")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"llm_output['priority'] must be one of {VALID_PRIORITIES}, got {priority!r}")

    config = load_env_config()
    alert_id = str(uuid.uuid4())
    model_used = derive_model_used(alert_context, llm_output)
    record = assemble_alert_record(alert_context, fused_score, llm_output, model_used, alert_id)

    receipt: dict[str, Any] = {
        "alert_id": alert_id,
        "priority": priority,
        "risk_score": record["risk_score"],
        "model_used": model_used,
    }

    try:
        receipt["rabbitmq"] = publish_to_rabbitmq(record, config)
    except Exception as exc:
        logger.exception("[alert_dispatcher] RabbitMQ publish failed")
        receipt["rabbitmq"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        receipt["influxdb"] = write_influx_point(record, config)
    except Exception as exc:
        logger.exception("[alert_dispatcher] InfluxDB write failed")
        receipt["influxdb"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        receipt["elasticsearch"] = write_elasticsearch_doc(record, config)
    except Exception as exc:
        logger.exception("[alert_dispatcher] Elasticsearch write failed")
        receipt["elasticsearch"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        if priority == "HIGH":
            receipt["notification"] = notify_line.send_line_alert(record, config)
        elif priority == "MEDIUM":
            receipt["notification"] = notify_gmail.send_immediate_email(record, config)
        else:
            receipt["notification"] = accumulate_low_priority_digest(record)
    except Exception as exc:
        logger.exception("[alert_dispatcher] notification step failed")
        receipt["notification"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    logger.info(
        "[alert_dispatcher] dispatched alert_id=%s priority=%s model_used=%s rabbitmq=%s influxdb=%s elasticsearch=%s notification=%s",
        alert_id, priority, model_used,
        receipt["rabbitmq"]["status"], receipt["influxdb"]["status"], receipt["elasticsearch"]["status"], receipt["notification"]["status"],
    )
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="alert_dispatcher.py -- SOC alert dispatch / LOW-digest flush")
    parser.add_argument("--flush-digest", action="store_true", help="flush the LOW-priority SQLite digest and send it via Gmail")
    parser.add_argument("--demo", action="store_true", help="dispatch the HIGH-priority worked-example fixture (tests/fixtures/) as a smoke test")
    args = parser.parse_args()

    if args.flush_digest:
        result = flush_low_priority_digest()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.demo:
        fixtures_dir = ROOT / "tests" / "fixtures"
        alert_context = json.loads((fixtures_dir / "alert_context_high_portscan.json").read_text(encoding="utf-8"))
        llm_output = json.loads((fixtures_dir / "llm_output_high_portscan.json").read_text(encoding="utf-8"))
        fused_score = alert_context["fused_score"]
        result = dispatch_alert(alert_context, fused_score, llm_output)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        parser.print_help()
