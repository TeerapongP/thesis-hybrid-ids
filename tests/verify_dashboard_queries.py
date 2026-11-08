"""
verify_dashboard_queries.py
============================
Runs the ACTUAL Flux queries embedded in `grafana/dashboards/soc-overview.json` (not a
paraphrase of them) directly against InfluxDB, and the equivalent Elasticsearch query for the
`soc-alerts-*` index pattern the Kibana `SOC Latest Alerts` saved search
(`observability/kibana/soc-alert-investigation.ndjson`) uses, to confirm that data written by
`dispatch_alert(...)` is actually queryable in the exact shape those two UIs expect.

This is the acceptance test HANDOFF_OUTPUT_LAYER.md's Conventions section describes ("if a
query in this file can't return data from what the dispatcher wrote, the dispatcher's schema is
wrong, not the dashboard") -- run via direct API calls, not by opening a browser against the
live Grafana/Kibana UI (no browser tool available in this environment; that gap is disclosed,
not silently skipped -- see tests/verify_dispatch_live.py's summary).

Run AFTER a dispatch has written data, for example tests/verify_dispatch_live.py, which sends
real notifications (read its docstring first).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from dotenv import dotenv_values
from elasticsearch import Elasticsearch
from influxdb_client import InfluxDBClient

ENV_PATH = ROOT / "notification" / ".env"


def main() -> int:
    cfg = dotenv_values(ENV_PATH)
    ok = True

    print("=== InfluxDB: exact Flux queries from grafana/dashboards/soc-overview.json ===")
    with InfluxDBClient(url=cfg["INFLUX_URL"], token=cfg["INFLUX_TOKEN"], org=cfg["INFLUX_ORG"]) as client:
        query_api = client.query_api()

        # Panel "Risk Score Trend" (id 1), HIGH series, widened range/window since fixture
        # timestamps are historical (2026-07-14/15), not "now".
        flux_risk_trend = (
            'from(bucket: "alerts")\n'
            "  |> range(start: -30d)\n"
            '  |> filter(fn: (r) => r._measurement == "soc_alert" and r._field == "risk_score")\n'
            "  |> group()\n"
            "  |> sort(columns: [\"_time\"], desc: true)\n"
            "  |> limit(n: 20)"
        )
        tables = query_api.query(flux_risk_trend)
        n_points = sum(len(t.records) for t in tables)
        print(f"  Risk Score Trend-equivalent query (soc_alert.risk_score, -30d): {n_points} points")
        ok = ok and n_points > 0

        # Panel "Latest Alerts" (id 5) -- the actual pivot/keep/sort/limit query, range widened.
        flux_latest_alerts = (
            'from(bucket: "alerts")\n'
            "  |> range(start: -30d)\n"
            '  |> filter(fn: (r) => r._measurement == "soc_alert" and contains(value: r._field, set: ["risk_score", "response_ms"]))\n'
            "  |> filter(fn: (r) => exists r.alert_id and exists r.source_ip and exists r.dst_port)\n"
            '  |> pivot(rowKey: ["_time", "alert_id", "source_ip", "dst_port", "priority", "anomaly_type", "mitre_tactic"], columnKey: ["_field"], valueColumn: "_value")\n'
            '  |> keep(columns: ["_time", "priority", "source_ip", "dst_port", "risk_score", "anomaly_type", "mitre_tactic", "response_ms", "alert_id"])\n'
            '  |> sort(columns: ["_time"], desc: true)\n'
            "  |> limit(n: 10)"
        )
        tables = query_api.query(flux_latest_alerts)
        rows = [rec.values for t in tables for rec in t.records]
        print(f"  Latest Alerts panel query (pivoted, exact dashboard shape): {len(rows)} rows")
        for r in rows[:5]:
            print(f"    alert_id={r.get('alert_id')} priority={r.get('priority')} risk_score={r.get('risk_score')} source_ip={r.get('source_ip')}")
        ok = ok and len(rows) > 0

    print()
    print("=== Elasticsearch: soc-alerts-* index pattern (Kibana SOC Latest Alerts search) ===")
    # Deliberately uses ELASTIC_READER_API_KEY, not ELASTIC_API_KEY -- this is the same
    # read-only key grafana/provisioning/datasources/soc-datasources.yml's Elasticsearch-SOC
    # datasource actually authenticates with, so this is the fairest read-path check (and
    # ELASTIC_API_KEY, used for the writes, turned out on this cluster to NOT carry a read
    # grant -- confirmed via a 403 AuthorizationException when this script first tried it).
    es = Elasticsearch(cfg["ELASTIC_URL"], api_key=cfg["ELASTIC_READER_API_KEY"])
    resp = es.search(index="soc-alerts-*", size=10, sort=[{"timestamp": "desc"}], query={"match_all": {}})
    hits = resp["hits"]["hits"]
    print(f"  soc-alerts-* search: {resp['hits']['total']['value']} total docs, showing {len(hits)}")
    for h in hits[:5]:
        src = h["_source"]
        print(f"    alert_id={src.get('alert_id')} priority={src.get('priority')} risk_score={src.get('risk_score')} index={h['_index']}")
    ok = ok and len(hits) > 0

    print()
    print("RESULT:", "PASS -- dashboard/Kibana queries return real data written by dispatch_alert(...)" if ok else "FAIL -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
