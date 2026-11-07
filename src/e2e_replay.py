"""e2e_replay.py -- replay a flow event through Kafka, enrichment, detection, explanation and
notification, so the ingestion half and the detection half of the system finally meet.

Until this module `src/services/enrich.py` had no producer and `src/pipeline.py` read the
feature parquet directly, so the two halves had never run in one chain. The chain here is

    parquet -> Kafka KAFKA_INPUT_TOPIC -> enrich.py (unmodified) -> KAFKA_OUTPUT_TOPIC
            -> fusion scoring -> RAG -> LLM -> dispatch_alert -> LINE / InfluxDB / Elasticsearch

WHAT THIS EARNS, AND WHAT IT DOES NOT. A replayed event is a pre-extracted CICFlowMeter
flow: it carries the 94 model features in its payload. That is deliberate. Publishing only
identifiers and having the scorer look the row up in the parquet would leave the Kafka leg
decorative and the two halves still unmet. The cost is that nothing here turns a raw
firewall log into features, and this repository has no such stage. So the claim this runner
supports is "end to end from a replayed flow event through enrichment, detection,
explanation and notification". It does not support "end to end from raw traffic" or "from
ingestion", and a deployment would need a feature-extraction stage between the log and the
model.

TWO INTERPRETERS, ONE CHAIN. `kafka` imports only in the uv .venv. `pandas` and `pyarrow`
import only in Anaconda. The chain therefore crosses interpreters at the Kafka seam, the
same way src/pipeline.py crosses at the score/LLM seam: the coordinator runs in the .venv and
shells out to Anaconda for the stages that declare it.

    uv run python -m src.e2e_replay --check
    uv run python -m src.e2e_replay --label PortScan --dry-run
    uv run python -m src.e2e_replay --row 44999 --dispatch
    uv run python -m src.e2e_replay --from score          # retry a late stage, keep the rest

THE ENRICHMENT SERVICE IS NOT MANAGED HERE. `enrich.py` is a long-running consumer the
operator starts in its own terminal:

    uv run python -m src.services.enrich

This runner never starts or stops it. A background consumer started by a script and then
lost is how orphaned processes accumulated on this project. If it is not running, the
`consume` stage fails and says so, and `--from consume` resumes once it is.

DRY RUN IS THE DEFAULT. `--dispatch` is the only way to call dispatch_alert, which sends a
real LINE broadcast (HIGH priority) and writes InfluxDB and Elasticsearch.

TIMESTAMPS ARE THE FLOW'S OWN. Replayed CICIDS-2017 flows are dated 2017-07-05, so
InfluxDB points land in 2017 and Elasticsearch documents in soc-alerts-2017.07.05. That is
deliberate: stamping wall-clock would assert that 2017 traffic happened today. A Grafana or
Kibana view over recent time will look empty even though the write succeeded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src import pipeline

ROOT = pipeline.ROOT
ML_PYTHON = pipeline.ML_PYTHON
ANACONDA, VENV = "anaconda", "venv"

EVENTS_FILE = ROOT / "reports/e2e_replay_events.jsonl"
RECEIPT_FILE = ROOT / "reports/e2e_produce_receipt.json"
ENRICHED_FILE = ROOT / "reports/e2e_enriched_events.jsonl"
LAST_RUN = ROOT / "reports/e2e_last_run.json"

DEFAULT_CONSUME_TIMEOUT_S = 120.0
ENRICH_START_HINT = "uv run python -m src.services.enrich"
REQUIRED_MODEL_FILES = (
    "models/random_forest_unified.pkl",
    "models/isolation_forest_unified.pkl",
    "models/autoencoder_unified.pt",
    "models/autoencoder_unified_config.json",
)


class StageError(Exception):
    """A stage cannot proceed or did not produce what it promised."""


# ----------------------------------------------------------------------------
# Events: the payload that crosses the Kafka seam
# ----------------------------------------------------------------------------
def build_event(
    run_id: str,
    n: int,
    ident_row: Mapping[str, Any],
    row_index: int,
    label: str,
    feature_names: Sequence[str],
    feature_values: Sequence[float],
) -> dict[str, Any]:
    """One replay event. `src_ip`/`dest_ip` use the spellings in enrich.IP_FIELDS."""
    if len(feature_names) != len(feature_values):
        raise ValueError(f"{len(feature_names)} feature names but {len(feature_values)} values")
    return {
        "event_id": f"e2e-{run_id}-{n}",
        "timestamp": pipeline._iso(ident_row["Timestamp"]),
        "src_ip": str(ident_row["Source IP"]),
        "dest_ip": str(ident_row["Destination IP"]),
        "dest_port": int(ident_row["Destination Port"]),
        "protocol": int(ident_row["Protocol"]),
        "test_row_index": int(row_index),
        "label": str(label),
        "features": {name: float(v) for name, v in zip(feature_names, feature_values)},
    }


def dumps_event(event: Mapping[str, Any]) -> str:
    """Compact JSON on one line. NaN is rejected: it is not valid JSON and would reach a
    consumer that parses strictly as a different failure far from its cause."""
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [dumps_event(r) for r in rows]
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return len(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def vectorize(features: Mapping[str, float], order: Sequence[str]) -> list[float]:
    """Feature values in model-input order, selected by NAME.

    Dict order is never trusted: a reordered payload would feed every model the right
    numbers in the wrong columns and still score without an error.
    """
    missing = [c for c in order if c not in features]
    extra = sorted(set(features) - set(order))
    if missing or extra:
        raise ValueError(f"feature set mismatch: missing {missing[:5]}, unexpected {extra[:5]}")
    return [float(features[c]) for c in order]


def verify_additive(raw: Mapping[str, Any], enriched: Mapping[str, Any]) -> list[str]:
    """Problems with the claim that enrichment only adds an `enrichment` object."""
    problems = [f"field {k!r} missing after enrichment" if k not in enriched
                else f"field {k!r} changed by enrichment"
                for k, v in raw.items() if k not in enriched or enriched[k] != v]
    problems += [f"unexpected new field {k!r}" for k in enriched if k not in raw and k != "enrichment"]
    if "enrichment" not in enriched:
        problems.append("no enrichment object")
    return problems


def collect_enriched(
    expected: Sequence[str],
    poll: Callable[[float], Iterable[bytes]],
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Poll until every expected event_id has been seen or the deadline passes.

    Returns (found, missing). Unrelated messages and a repeated event_id are ignored, so the
    result is the same whether the topic holds one run or a hundred.
    """
    want = set(expected)
    found: dict[str, dict[str, Any]] = {}
    deadline = clock() + timeout_s
    while len(found) < len(want):
        remaining = deadline - clock()
        if remaining <= 0:
            break
        for raw in poll(min(1.0, remaining)):
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            eid = event.get("event_id") if isinstance(event, dict) else None
            if eid in want and eid not in found:
                found[eid] = event
    return found, [e for e in expected if e not in found]


def resolve_enriched(
    expected: Sequence[str],
    raw_by_id: Mapping[str, Mapping[str, Any]],
    poll: Callable[[float], Iterable[bytes]],
    timeout_s: float,
    topic: str,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """The enriched events in `expected` order, or a StageError saying what is wrong.

    A partial set is never returned: scoring four of five events would report a chain that
    did not complete as one that did.
    """
    found, missing = collect_enriched(expected, poll, timeout_s, clock)
    if missing:
        raise StageError(
            f"{len(missing)} of {len(expected)} event(s) never appeared on {topic} within "
            f"{timeout_s:g}s: {missing}. enrich.py produces them and this runner does not start "
            f"it. In another terminal run `{ENRICH_START_HINT}`, then retry with `--from consume`; "
            f"the produced events are still on the broker."
        )
    problems = [f"{eid}: {p}" for eid in expected for p in verify_additive(raw_by_id[eid], found[eid])]
    if problems:
        raise StageError("enrichment was not purely additive: " + "; ".join(problems))
    return [found[eid] for eid in expected]


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def export_stage(args) -> None:
    import numpy as np
    import pandas as pd

    from src.features import build_unified_population as BUP

    test = pd.read_parquet(pipeline.UNIFIED_TEST)
    labels = test["Label"].tolist()
    try:
        pos_idx = pipeline.select_rows(labels, rows=args.row, label=args.label,
                                       random_n=args.random, seed=args.seed)
    except ValueError as exc:
        raise StageError(str(exc)) from exc
    feat_cols = pipeline.feature_columns(test)
    x = test.iloc[pos_idx][feat_cols].to_numpy(dtype=np.float64)

    ident_cols = ["Label", "Source IP", "Destination IP", "Destination Port", "Protocol", "Timestamp"]
    _, _, raw_test = BUP.split_population(pd.read_parquet(BUP.SOURCE, columns=ident_cols))
    ident = pipeline.recover_identifiers(raw_test, pos_idx, labels)

    run_id = args.run_id or time.strftime("%Y%m%d%H%M%S")
    events = [
        build_event(run_id, n, ident.iloc[n], row_index, labels[row_index], feat_cols, x[n])
        for n, row_index in enumerate(pos_idx)
    ]
    write_jsonl(EVENTS_FILE, events)
    print(f"[export] run {run_id}: {len(events)} event(s) from test rows {pos_idx}", flush=True)


def _enrich_config():
    from src.services import enrich

    try:
        return enrich.load_config()
    except RuntimeError as exc:
        raise StageError(str(exc)) from exc


def produce_stage(args) -> None:
    try:
        from kafka import KafkaProducer
        from kafka.errors import KafkaError
    except ImportError as exc:
        raise StageError("kafka-python is only in the uv .venv; run `uv run python -m src.e2e_replay`") from exc

    cfg = _enrich_config()
    events = load_jsonl(EVENTS_FILE)
    try:
        producer = KafkaProducer(bootstrap_servers=cfg.bootstrap, acks="all", max_block_ms=15000)
        futures = [
            producer.send(cfg.input_topic, key=e["event_id"].encode(), value=dumps_event(e).encode("utf-8"))
            for e in events
        ]
        producer.flush(timeout=30)
        for f in futures:
            f.get(timeout=15)
        producer.close()
    except KafkaError as exc:
        raise StageError(f"publishing to {cfg.input_topic} on {cfg.bootstrap} failed: {exc!r}") from exc

    receipt = {
        "topic": cfg.input_topic, "count": len(events),
        "event_ids": [e["event_id"] for e in events],
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    RECEIPT_FILE.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[produce] {len(events)} event(s) acknowledged on {cfg.input_topic}", flush=True)


def consume_stage(args) -> None:
    try:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError
    except ImportError as exc:
        raise StageError("kafka-python is only in the uv .venv; run `uv run python -m src.e2e_replay`") from exc

    cfg = _enrich_config()
    expected = json.loads(RECEIPT_FILE.read_text(encoding="utf-8"))["event_ids"]
    raw_by_id = {e["event_id"]: e for e in load_jsonl(EVENTS_FILE)}
    try:
        # No group: reading from the start and filtering by event_id makes a retry
        # idempotent and leaves no committed offset behind to confuse enrich.py's own group.
        consumer = KafkaConsumer(
            cfg.output_topic, bootstrap_servers=cfg.bootstrap, group_id=None,
            auto_offset_reset="earliest", enable_auto_commit=False,
        )
    except KafkaError as exc:
        raise StageError(f"cannot read {cfg.output_topic} on {cfg.bootstrap}: {exc!r}") from exc

    def poll(timeout_s: float) -> list[bytes]:
        batches = consumer.poll(timeout_ms=int(timeout_s * 1000))
        return [m.value for msgs in batches.values() for m in msgs]

    try:
        enriched = resolve_enriched(expected, raw_by_id, poll, args.consume_timeout, cfg.output_topic)
    finally:
        consumer.close()
    write_jsonl(ENRICHED_FILE, enriched)
    print(f"[consume] {len(enriched)} enriched event(s) from {cfg.output_topic}", flush=True)


def model_feature_order() -> list[str]:
    """The model-input column order, from the schema of the file the models were trained on."""
    import pyarrow.parquet as pq

    empty = pq.ParquetFile(pipeline.UNIFIED_TEST).schema_arrow.empty_table().to_pandas()
    return pipeline.feature_columns(empty)


def score_stage(args) -> None:
    import numpy as np

    from src.modeling import fusion_layer
    from src.rag.rag_retrieval import RagRetriever

    events = load_jsonl(ENRICHED_FILE)
    order = model_feature_order()
    try:
        x = np.asarray([vectorize(e["features"], order) for e in events], dtype=np.float64)
    except ValueError as exc:
        raise StageError(str(exc)) from exc
    print(f"[score] {len(events)} event(s), {len(order)} features selected by name", flush=True)

    scored = pipeline.score_matrix(pipeline.load_models(), x)
    retriever = RagRetriever()
    alerts = []
    for i, event in enumerate(events):
        ident = {
            "Timestamp": dt.datetime.fromisoformat(event["timestamp"]),
            "Source IP": event["src_ip"], "Destination IP": event["dest_ip"],
            "Destination Port": event["dest_port"],
        }
        entry = pipeline.alert_entry(event["test_row_index"], event["label"], ident, scored, i,
                                     retriever, args.top_k)
        entry["event_id"] = event["event_id"]
        entry["enrichment"] = event["enrichment"]
        alerts.append(entry)

    payload = {
        "stage": "score",
        "source": "e2e_replay",
        "n_features": len(order),
        "fusion_strategy": fusion_layer.CHOSEN_STRATEGY,
        "score_latency_ms_per_row": scored["latency_s_per_row"] * 1000,
        "shap": False,
        "alerts": alerts,
    }
    LAST_RUN.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[score] wrote {LAST_RUN}", flush=True)


def notify_stage(args) -> None:
    payload = json.loads(LAST_RUN.read_text(encoding="utf-8"))
    dry_run = not args.dispatch
    try:
        pipeline.llm_stage(payload)
    except RuntimeError as exc:
        raise StageError(str(exc)) from exc
    pipeline.dispatch_stage(payload, dry_run)
    payload["dry_run"] = dry_run
    LAST_RUN.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[notify] {'dry run, nothing sent' if dry_run else 'dispatched'}", flush=True)


# ----------------------------------------------------------------------------
# The stage registry and the coordinator that walks it
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    key: str
    purpose: str
    interpreter: str
    needs: Path | None
    produces: Path
    run: Callable[[Any], None]

    @property
    def precondition(self) -> str:
        return "none" if self.needs is None else f"{self.needs.name} exists and is non-empty"


STAGES: tuple[Stage, ...] = (
    Stage("export", "pick test rows and write replay events with their 94 features", ANACONDA,
          None, EVENTS_FILE, export_stage),
    Stage("produce", "publish the events to KAFKA_INPUT_TOPIC", VENV,
          EVENTS_FILE, RECEIPT_FILE, produce_stage),
    Stage("consume", "collect the matching events enrich.py published to KAFKA_OUTPUT_TOPIC", VENV,
          RECEIPT_FILE, ENRICHED_FILE, consume_stage),
    Stage("score", "fuse and retrieve MITRE context from each event's own features", ANACONDA,
          ENRICHED_FILE, LAST_RUN, score_stage),
    Stage("notify", "Thai explanation, then dispatch (or dry run)", VENV,
          LAST_RUN, LAST_RUN, notify_stage),
)


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def plan(stages: Sequence[Stage], start: str | None) -> list[Stage]:
    if start is None:
        return list(stages)
    keys = [s.key for s in stages]
    if start not in keys:
        raise StageError(f"unknown stage {start!r}; stages are {keys}")
    return list(stages[keys.index(start):])


@dataclass(frozen=True)
class StageResult:
    key: str
    seconds: float
    ok: bool
    detail: str = ""


def run_chain(
    stages: Sequence[Stage],
    start: str | None,
    args,
    execute: Callable[[Stage, Any], None],
    clock: Callable[[], float] = time.perf_counter,
) -> list[StageResult]:
    """Walk the registry from `start`, checking each stage's input before and output after.

    Stops at the first failure and returns what ran, the failing stage last.
    """
    results: list[StageResult] = []
    for stage in plan(stages, start):
        started = clock()
        try:
            if stage.needs is not None and not nonempty(stage.needs):
                raise StageError(f"{stage.key} needs {stage.needs}, which is missing or empty; "
                                 f"start from an earlier stage")
            # A leftover artifact from an earlier run would let a stage that wrote nothing
            # pass the check below. A stage that updates its own input in place cannot be
            # cleared first, so it is trusted to raise on failure.
            if stage.produces != stage.needs:
                stage.produces.unlink(missing_ok=True)
            execute(stage, args)
            if not nonempty(stage.produces):
                raise StageError(f"{stage.key} finished but {stage.produces} is missing or empty")
        except (StageError, subprocess.CalledProcessError) as exc:
            results.append(StageResult(stage.key, clock() - started, False, str(exc)))
            return results
        results.append(StageResult(stage.key, clock() - started, True))
    return results


def forward_args(args) -> list[str]:
    out = ["--seed", str(args.seed), "--top-k", str(args.top_k),
           "--consume-timeout", str(args.consume_timeout)]
    for r in args.row or []:
        out += ["--row", str(r)]
    if args.label:
        out += ["--label", args.label]
    if args.random:
        out += ["--random", str(args.random)]
    if args.run_id:
        out += ["--run-id", args.run_id]
    return out


def execute(stage: Stage, args) -> None:
    print(f"[e2e] {stage.key} ({stage.interpreter}): {stage.purpose}", flush=True)
    if stage.interpreter == VENV:
        stage.run(args)
        return
    cmd = [ML_PYTHON, "-m", "src.e2e_replay", "--run-stage", stage.key, *forward_args(args)]
    subprocess.run(cmd, cwd=ROOT, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


# ----------------------------------------------------------------------------
# Preconditions (--check): read-only, starts and stops nothing
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(checks: Sequence[Check]) -> list[CheckResult]:
    """Every check reports PASS or FAIL. One that raises is a FAIL carrying the reason."""
    out = []
    for check in checks:
        try:
            out.append(CheckResult(check.name, True, check.run()))
        except Exception as exc:
            out.append(CheckResult(check.name, False, f"{type(exc).__name__}: {exc}"))
    return out


def check_enrich_config() -> str:
    cfg = _enrich_config()
    return f"input={cfg.input_topic} output={cfg.output_topic} group={cfg.group_id}"


def _admin(bootstrap: str):
    from kafka.admin import KafkaAdminClient

    return KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=5000,
                            api_version_auto_timeout_ms=5000)


def check_kafka() -> str:
    cfg = _enrich_config()
    admin = _admin(cfg.bootstrap)
    try:
        topics = set(admin.list_topics())
    finally:
        admin.close()
    state = {t: ("present" if t in topics else "absent") for t in (cfg.input_topic, cfg.output_topic)}
    return f"broker {cfg.bootstrap} reachable; topics {state}"


def check_enrich_running() -> str:
    cfg = _enrich_config()
    admin = _admin(cfg.bootstrap)
    try:
        info = admin.describe_consumer_groups([cfg.group_id])[0]
    finally:
        admin.close()
    members = len(info.members)
    if members == 0:
        raise RuntimeError(f"group {cfg.group_id} has no active member (state {info.state}); "
                           f"start it with `{ENRICH_START_HINT}`")
    return f"group {cfg.group_id} has {members} active member(s), state {info.state}"


def check_models(root: Path = ROOT) -> str:
    missing = [f for f in REQUIRED_MODEL_FILES if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing}")
    return f"{len(REQUIRED_MODEL_FILES)} model files present"


def check_ollama() -> str:
    from src.llm import llm_explain

    config = llm_explain.load_env_config()
    up, detail = pipeline.ollama_status(config["base_url"])
    if not up:
        raise ConnectionError(f"{config['base_url']} unreachable ({detail}); run `ollama serve`")
    if config["model"] not in detail:
        raise LookupError(f"{config['base_url']} is up but model {config['model']} is not pulled")
    return f"{config['base_url']} up with {config['model']}"


def check_rabbitmq_vhost() -> str:
    import pika

    from src.notifications import alert_dispatcher

    config = alert_dispatcher.load_env_config()
    pika.BlockingConnection(pika.URLParameters(config["rabbitmq_url"])).close()
    return f"connected to vhost {config['rabbitmq_vhost']}"


def check_sink_ports() -> str:
    from src.notifications import alert_dispatcher

    config = alert_dispatcher.load_env_config()
    down = []
    for name in ("influx_url", "elastic_url"):
        parts = urllib.parse.urlsplit(config[name])
        try:
            socket.create_connection((parts.hostname, parts.port or 80), timeout=3).close()
        except OSError as exc:
            down.append(f"{name}: {exc}")
    if down:
        raise ConnectionError("; ".join(down))
    return "InfluxDB and Elasticsearch accept connections"


def check_line() -> str:
    proc = subprocess.run([sys.executable, "-m", "src.notifications.line_preflight"], cwd=ROOT,
                          capture_output=True, text=True, timeout=30)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    tail = lines[-1] if lines else "no output"
    if proc.returncode != 0:
        raise RuntimeError(tail)
    return tail


def default_checks() -> list[Check]:
    return [
        Check("enrich config + topic names", check_enrich_config),
        Check("kafka broker", check_kafka),
        Check("enrich.py running", check_enrich_running),
        Check("model files", check_models),
        Check("ollama + model", check_ollama),
        Check("rabbitmq vhost (--dispatch)", check_rabbitmq_vhost),
        Check("influxdb + elasticsearch (--dispatch)", check_sink_ports),
        Check("line channel (--dispatch)", check_line),
    ]


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def alert_completed(alert: Mapping[str, Any]) -> bool:
    """The event came back enriched through Kafka and reached a dispatch outcome."""
    d = alert.get("dispatch")
    if not alert.get("enrichment") or not d:
        return False
    if d.get("status") in ("dry_run", "suppressed"):
        return True
    if d.get("status") == "skipped":
        return False
    sinks = ("rabbitmq", "influxdb", "elasticsearch", "notification")
    return all(d.get(s, {}).get("status") != "error" for s in sinks)


def _enrichment_cell(enrichment: Mapping[str, Any] | None) -> str:
    if not enrichment:
        return "none"
    if enrichment.get("error"):
        return "error"
    if enrichment.get("matched"):
        return f"matched ({len(enrichment.get('indicators') or [])} indicator(s))"
    return f"no match ({len(enrichment.get('observables_checked') or [])} public IP(s) checked)"


def _dispatch_cell(d: Mapping[str, Any] | None) -> str:
    if not d:
        return "not reached"
    if d.get("status") in ("dry_run", "suppressed", "skipped"):
        return d["status"].replace("_", " ")
    return " ".join(f"{s}={d.get(s, {}).get('status')}"
                    for s in ("rabbitmq", "influxdb", "elasticsearch", "notification"))


def render_e2e_summary(payload: Mapping[str, Any], results: Sequence[StageResult]) -> str:
    alerts = payload["alerts"]
    L = ["", "=" * 78, "E2E REPLAY SUMMARY", "=" * 78]
    for a in alerts:
        ctx = a["alert_context"]
        out = a.get("llm_output") or {}
        priority = out.get("priority") or ("suppressed" if a.get("suppressed") else "-")
        L += [f"row {a['test_row_index']}  {a['true_label']}  {a['event_id']}",
              f"    model_used={ctx['model_used']}  fused_score={ctx['fused_score']:.4f}  priority={priority}",
              f"    enrichment: {_enrichment_cell(a.get('enrichment'))}",
              f"    dispatch:   {_dispatch_cell(a.get('dispatch'))}"]
    done = sum(alert_completed(a) for a in alerts)
    L += ["", "stage wall time"]
    L += [f"    {r.key:<8} {r.seconds:7.2f}s  {'ok' if r.ok else 'FAILED'}" for r in results]
    L += ["", f"{done} of {len(alerts)} event(s) completed the whole chain "
              f"(enriched through Kafka, scored, explained, dispatched or suppressed)"]
    days = sorted({a["alert_context"]["timestamp"][:10] for a in alerts})
    indices = [f"soc-alerts-{d.replace('-', '.')}" for d in days]
    L += ["", f"note: event time is the flow's own capture time, not today's. InfluxDB points are dated "
              f"{days} and Elasticsearch documents go to {indices}, so a Grafana or Kibana view "
              f"over recent time looks empty."]
    if payload.get("dry_run", True):
        L.append("note: dry run. dispatch_alert was not called and nothing was sent.")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    keys = [s.key for s in STAGES]
    p = argparse.ArgumentParser(
        prog="python -m src.e2e_replay",
        description="Replay flow events through Kafka, enrichment, detection, explanation and dispatch.",
    )
    p.add_argument("--row", type=int, action="append", help="test-split row index (repeatable)")
    p.add_argument("--label", help='pick rows with this Label, e.g. "PortScan"')
    p.add_argument("--random", type=int, metavar="N", help="pick N random rows")
    p.add_argument("--seed", type=int, default=pipeline.SEED)
    p.add_argument("--top-k", type=int, default=pipeline.DEFAULT_TOP_K, help="RAG hits per alert")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="run every layer but send nothing (DEFAULT)")
    p.add_argument("--dispatch", action="store_true",
                   help="actually call dispatch_alert -- sends a real LINE broadcast for HIGH")
    p.add_argument("--from", dest="from_stage", choices=keys, help="start at this stage")
    p.add_argument("--consume-timeout", type=float, default=DEFAULT_CONSUME_TIMEOUT_S,
                   help="seconds to wait for enriched events (default %(default)s)")
    p.add_argument("--check", action="store_true", help="verify preconditions, send nothing")
    p.add_argument("--full", action="store_true", help="also print pipeline.render_summary per alert")
    p.add_argument("--run-id", help="event_id prefix (default: timestamp)")
    p.add_argument("--run-stage", choices=keys, help="internal: run one stage in this interpreter")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)

    if args.run_stage:
        stage = next(s for s in STAGES if s.key == args.run_stage)
        try:
            stage.run(args)
        except StageError as exc:
            print(f"[{stage.key}] {exc}", file=sys.stderr)
            return 1
        return 0

    if args.check:
        results = run_checks(default_checks())
        for r in results:
            print(f"{'PASS' if r.ok else 'FAIL'}  {r.name:<38} {r.detail}")
        return 0 if all(r.ok for r in results) else 1

    results = run_chain(STAGES, args.from_stage, args, execute)
    if not results[-1].ok:
        failed = results[-1]
        print(f"[e2e] stage {failed.key} FAILED after {failed.seconds:.1f}s: {failed.detail}",
              file=sys.stderr)
        print(f"[e2e] retry from there with --from {failed.key}", file=sys.stderr)
        return 1

    payload = json.loads(LAST_RUN.read_text(encoding="utf-8"))
    if args.full:
        print(pipeline.render_summary(payload))
    print(render_e2e_summary(payload, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
