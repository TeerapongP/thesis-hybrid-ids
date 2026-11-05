"""
enrich.py
=========
OpenCTI enrichment service. Consumes Kafka `pa5220.raw`, enriches each event against OpenCTI,
and publishes to `pa5220.enriched` (which `observability/logstash/pipeline/soc.conf` already
indexes).

    python -m src.services.enrich --dry-run            # stubbed OpenCTI + in-memory broker
    python -m src.services.enrich --dry-run --input events.jsonl
    python -m src.services.enrich                      # live: Kafka + OpenCTI from root .env

Output contract: the enriched event is the raw event plus ONE additive `enrichment` object
(`matched`, `observables_checked`, `indicators`, `related`, `enriched_at`, `source`, and an
`error` string only when OpenCTI could not be reached). No raw field is ever changed or removed,
so `pa5220.raw` and `pa5220.enriched` stay diffable in Elasticsearch.

Operational properties:
  * Idempotent. Every field of the output is a pure function of (raw event, OpenCTI answer,
    Kafka message timestamp). `enriched_at` comes from the message timestamp (stable across
    redelivery), then the event's own timestamp, and only then the wall clock. Offsets are
    committed only after the publish is acknowledged. `enrichment.source` is provenance
    ("opencti" vs "cache") and can legitimately differ on a redelivery that hits the cache;
    compare enriched events with `source` excluded.
  * Degrades, never blocks. An OpenCTI failure publishes the event with `matched: false` and
    `enrichment.error`. A circuit breaker stops paying the request timeout per event while
    OpenCTI is down. Failures are never cached.
  * Bounded cache keyed by observable (`OPENCTI_CACHE_SIZE`, LRU, with a TTL so a stale
    negative answer cannot hide a newly published indicator forever).
  * Pure core. `extract_observables`, `match_indicators`, `build_enrichment` and `enrich_event`
    take parsed types and need no broker or network. Kafka and OpenCTI are reached lazily, only
    on the live path.

Config is read from the repo-root `.env` (process environment wins). Secrets are never logged.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_ENV_PATH = ROOT / ".env"

logger = logging.getLogger("enrich")

# Raw-event field names that may carry an IP observable. PA-5220 exports differ by pipeline
# stage, so accept the common spellings. Extend here, not in the logic.
IP_FIELDS: tuple[str, ...] = (
    "src_ip", "source_ip", "src", "Source Address", "Source IP",
    # `dest_ip` and the capitalized `Destination Address` were missing while their source-side
    # counterparts were present, so a destination address spelled either way was silently
    # never looked up -- found by a live run whose dest_ip went unchecked. Lookup is keyed on
    # the exact field name, so each spelling has to be listed; the asymmetry is the bug.
    "dst_ip", "dest_ip", "destination_ip", "dst",
    "Destination Address", "Destination address", "Destination IP",
)

GRAPH_WALK_LIMIT = 25
_PATTERN_IP = re.compile(r"(?:ipv4|ipv6)-addr:value\s*=\s*'([^']+)'", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Domain types
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicator:
    id: str
    name: str
    pattern: str
    confidence: int | None
    valid_until: str | None

    def as_output(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "confidence": self.confidence,
                "valid_until": self.valid_until}


@dataclass(frozen=True)
class Related:
    id: str
    type: str
    name: str
    relationship: str

    def as_output(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "name": self.name,
                "relationship": self.relationship}


@dataclass(frozen=True)
class Lookup:
    """OpenCTI's answer for one observable. `indicators` are already pattern-verified."""
    indicators: tuple[Indicator, ...] = ()
    related: tuple[Related, ...] = ()


class OpenCTIError(Exception):
    """Any failure to get an answer from OpenCTI (unreachable, slow, malformed, rejected)."""


class OpenCTIClient(Protocol):
    def lookup(self, observable: str) -> Lookup: ...


@dataclass(frozen=True)
class Config:
    bootstrap: str
    input_topic: str
    output_topic: str
    group_id: str
    auto_offset: str
    connect_retries: int
    opencti_url: str
    opencti_token: str = field(repr=False)
    cache_size: int = 5000
    log_level: str = "INFO"


REQUIRED_KEYS = (
    "KAFKA_BOOTSTRAP", "KAFKA_INPUT_TOPIC", "KAFKA_OUTPUT_TOPIC", "KAFKA_GROUP_ID",
    "KAFKA_AUTO_OFFSET", "KAFKA_CONNECT_RETRIES", "OPENCTI_URL", "OPENCTI_TOKEN",
    "OPENCTI_CACHE_SIZE",
)


def load_config(env_path: Path = ROOT_ENV_PATH, environ: Mapping[str, str] | None = None) -> Config:
    """Fail fast, naming every missing key (never its value). Process env overrides the file."""
    from dotenv import dotenv_values

    raw: dict[str, Any] = dict(dotenv_values(env_path)) if env_path.exists() else {}
    raw.update(os.environ if environ is None else environ)
    missing = [k for k in REQUIRED_KEYS if not raw.get(k)]
    if missing:
        raise RuntimeError(f"enrich config is missing required keys: {missing}")
    try:
        retries = int(raw["KAFKA_CONNECT_RETRIES"])
        cache_size = int(raw["OPENCTI_CACHE_SIZE"])
    except ValueError as exc:
        raise RuntimeError("KAFKA_CONNECT_RETRIES and OPENCTI_CACHE_SIZE must be integers") from exc
    if retries < 0 or cache_size < 1:
        raise RuntimeError("KAFKA_CONNECT_RETRIES must be >= 0 and OPENCTI_CACHE_SIZE >= 1")
    return Config(
        bootstrap=raw["KAFKA_BOOTSTRAP"], input_topic=raw["KAFKA_INPUT_TOPIC"],
        output_topic=raw["KAFKA_OUTPUT_TOPIC"], group_id=raw["KAFKA_GROUP_ID"],
        auto_offset=raw["KAFKA_AUTO_OFFSET"], connect_retries=retries,
        opencti_url=raw["OPENCTI_URL"].rstrip("/"), opencti_token=raw["OPENCTI_TOKEN"],
        cache_size=cache_size, log_level=(raw.get("LOG_LEVEL") or "INFO").upper(),
    )


# ----------------------------------------------------------------------------
# Pure core
# ----------------------------------------------------------------------------
def extract_observables(event: Mapping[str, Any]) -> list[str]:
    """Distinct, normalized, globally routable IPs found in the event, in field order.
    Private, loopback, link-local, multicast and reserved addresses are skipped: threat-intel
    indicators never describe them, so querying is pure cost."""
    seen: dict[str, None] = {}
    for key in IP_FIELDS:
        value = event.get(key)
        if not isinstance(value, str):
            continue
        try:
            ip = ipaddress.ip_address(value.strip())
        except ValueError:
            continue
        if ip.is_global:
            seen.setdefault(str(ip), None)
    return list(seen)


def pattern_matches_observable(pattern: str, observable: str) -> bool:
    """True if a STIX pattern's ipv4/ipv6 value equals the observable or is a CIDR containing
    it. OpenCTI's full-text search is fuzzy, so every hit is re-verified here."""
    try:
        ip = ipaddress.ip_address(observable)
    except ValueError:
        return False
    for value in _PATTERN_IP.findall(pattern):
        try:
            if ip in ipaddress.ip_network(value.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def match_indicators(observable: str, candidates: Iterable[Indicator]) -> tuple[Indicator, ...]:
    return tuple(i for i in candidates if pattern_matches_observable(i.pattern, observable))


def parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_enrichment(
    observables: Sequence[str],
    lookups: Mapping[str, Lookup],
    sources: Mapping[str, str],
    enriched_at: str,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Merge per-observable lookups into the one `enrichment` object. Indicators and related
    nodes are de-duplicated and ordered by first appearance so output is deterministic."""
    indicators: dict[str, Indicator] = {}
    related: dict[tuple[str, str], Related] = {}
    for obs in observables:
        found = lookups.get(obs)
        if found is None:
            continue
        for ind in found.indicators:
            indicators.setdefault(ind.id, ind)
        for rel in found.related:
            related.setdefault((rel.id, rel.relationship), rel)
    used = {sources[o] for o in observables if o in lookups}
    out: dict[str, Any] = {
        "matched": bool(indicators),
        "observables_checked": list(observables),
        "indicators": [i.as_output() for i in indicators.values()],
        "related": [r.as_output() for r in related.values()],
        "enriched_at": enriched_at,
        "source": "cache" if used == {"cache"} else "opencti",
    }
    if errors:
        out["error"] = "; ".join(dict.fromkeys(errors))
    return out


LookupFn = Callable[[str], "tuple[Lookup, str]"]


def enrich_event(
    event: Mapping[str, Any],
    lookup: LookupFn,
    message_time: datetime | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Return a new dict: every raw key unchanged, plus `enrichment`.

    `lookup(observable)` returns (Lookup, source) or raises OpenCTIError. A failing observable
    is reported in `enrichment.error`; it never propagates."""
    observables = extract_observables(event)
    lookups: dict[str, Lookup] = {}
    sources: dict[str, str] = {}
    errors: list[str] = []
    for obs in observables:
        try:
            lookups[obs], sources[obs] = lookup(obs)
        except OpenCTIError as exc:
            errors.append(f"{obs}: {exc}")

    event_ts = parse_iso8601(event.get("timestamp"))
    stamp = message_time or event_ts or now()
    enrichment = build_enrichment(observables, lookups, sources, _iso(stamp), errors)

    out = dict(event)
    if "timestamp" not in out:
        out["timestamp"] = _iso(stamp)
    elif event_ts is None:
        enrichment["timestamp_warning"] = "raw timestamp is not ISO8601; Elasticsearch will not date this event"
    out["enrichment"] = enrichment
    return out


# ----------------------------------------------------------------------------
# Cache + circuit breaker (stateful wrapper around a client)
# ----------------------------------------------------------------------------
class CachedLookup:
    """LRU + TTL cache keyed by observable, with a circuit breaker around the client.
    Failures are never cached, so a recovered OpenCTI is used on the very next event."""

    def __init__(
        self,
        client: OpenCTIClient,
        max_size: int,
        ttl_seconds: float = 300.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._client = client
        self._max = max_size
        self._ttl = ttl_seconds
        self._threshold = breaker_threshold
        self._cooldown = breaker_cooldown
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, Lookup]] = OrderedDict()
        self._failures = 0
        self._open_until = 0.0
        self.queries = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __call__(self, observable: str) -> tuple[Lookup, str]:
        now = self._clock()
        hit = self._entries.get(observable)
        if hit is not None and now - hit[0] <= self._ttl:
            self._entries.move_to_end(observable)
            return hit[1], "cache"
        if now < self._open_until:
            raise OpenCTIError("circuit open after repeated OpenCTI failures")
        self.queries += 1
        try:
            result = self._client.lookup(observable)
        except OpenCTIError:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open_until = now + self._cooldown
            raise
        self._failures = 0
        self._entries[observable] = (now, result)
        self._entries.move_to_end(observable)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
        return result, "opencti"


# ----------------------------------------------------------------------------
# OpenCTI clients
# ----------------------------------------------------------------------------
_INDICATOR_QUERY = """
query Indicators($search: String, $first: Int) {
  indicators(search: $search, first: $first) {
    edges { node { id name pattern confidence valid_until } }
  }
}"""

_WALK_QUERY = """
query Walk($id: [String]!, $first: Int) {
  stixCoreRelationships(fromId: $id, first: $first) {
    edges { node {
      relationship_type
      to {
        ... on StixDomainObject { id entity_type }
        ... on IntrusionSet { name }
        ... on Malware { name }
        ... on AttackPattern { name }
        ... on Campaign { name }
        ... on ThreatActor { name }
      }
    } }
  }
}"""

WALK_TARGET_TYPES = frozenset({"Intrusion-Set", "Malware", "Attack-Pattern", "Campaign", "Threat-Actor"})


def parse_indicators(payload: Mapping[str, Any]) -> list[Indicator]:
    """Boundary: coerce OpenCTI's GraphQL `indicators` payload; skip malformed nodes."""
    edges = (((payload.get("data") or {}).get("indicators") or {}).get("edges")) or []
    out: list[Indicator] = []
    for edge in edges:
        node = (edge or {}).get("node") or {}
        if not isinstance(node.get("id"), str) or not isinstance(node.get("pattern"), str):
            continue
        conf = node.get("confidence")
        out.append(Indicator(
            id=node["id"], name=str(node.get("name") or ""), pattern=node["pattern"],
            confidence=conf if isinstance(conf, int) and not isinstance(conf, bool) else None,
            valid_until=node["valid_until"] if isinstance(node.get("valid_until"), str) else None,
        ))
    return out


def parse_related(payload: Mapping[str, Any]) -> list[Related]:
    edges = (((payload.get("data") or {}).get("stixCoreRelationships") or {}).get("edges")) or []
    out: list[Related] = []
    for edge in edges:
        node = (edge or {}).get("node") or {}
        target = node.get("to") or {}
        if target.get("entity_type") not in WALK_TARGET_TYPES or not isinstance(target.get("id"), str):
            continue
        out.append(Related(
            id=target["id"], type=target["entity_type"], name=str(target.get("name") or ""),
            relationship=str(node.get("relationship_type") or ""),
        ))
    return out


class HttpOpenCTIClient:
    """GraphQL over plain HTTP (stdlib only; pycti is not required). The query shapes target
    OpenCTI 5.12.x and have NOT been exercised against a live instance."""

    def __init__(self, url: str, token: str, timeout: float = 5.0) -> None:
        self._endpoint = url.rstrip("/") + "/graphql"
        self._token = token
        self._timeout = timeout

    def _post(self, query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        req = urllib.request.Request(
            self._endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise OpenCTIError(type(exc).__name__) from exc
        if not isinstance(payload, dict) or payload.get("errors"):
            raise OpenCTIError("OpenCTI returned a GraphQL error")
        return payload

    def lookup(self, observable: str) -> Lookup:
        found = match_indicators(
            observable,
            parse_indicators(self._post(_INDICATOR_QUERY, {"search": observable, "first": 20})),
        )
        related: dict[tuple[str, str], Related] = {}
        for ind in found:
            for rel in parse_related(self._post(_WALK_QUERY, {"id": ind.id, "first": GRAPH_WALK_LIMIT})):
                related.setdefault((rel.id, rel.relationship), rel)
        return Lookup(indicators=found, related=tuple(related.values()))


DRY_RUN_TABLE: dict[str, Lookup] = {
    "45.33.32.156": Lookup(
        indicators=(Indicator("indicator--stub-1", "Scanner 45.33.32.156",
                              "[ipv4-addr:value = '45.33.32.156']", 80, "2027-01-01T00:00:00Z"),),
        related=(Related("intrusion-set--stub-1", "Intrusion-Set", "StubGroup", "indicates"),
                 Related("attack-pattern--stub-1", "Attack-Pattern", "Network Service Discovery", "indicates")),
    ),
}

DRY_RUN_EVENTS: tuple[dict[str, Any], ...] = (
    {"timestamp": "2026-07-01T10:00:00Z", "src_ip": "45.33.32.156", "dst_ip": "10.0.0.5", "dst_port": 22, "action": "allow"},
    {"timestamp": "2026-07-01T10:00:01Z", "src_ip": "45.33.32.156", "dst_ip": "93.184.216.34", "dst_port": 443, "action": "allow"},
    {"src_ip": "10.1.2.3", "dst_ip": "10.0.0.5", "dst_port": 80, "action": "deny"},
)


class StubOpenCTIClient:
    """Offline stand-in used by --dry-run and tests."""

    def __init__(self, table: Mapping[str, Lookup] | None = None, fail: bool = False) -> None:
        self.table = dict(DRY_RUN_TABLE if table is None else table)
        self.fail = fail
        self.calls: list[str] = []

    def lookup(self, observable: str) -> Lookup:
        self.calls.append(observable)
        if self.fail:
            raise OpenCTIError("stub configured to fail")
        return self.table.get(observable, Lookup())


# ----------------------------------------------------------------------------
# Streaming loop (broker-agnostic: producer is duck-typed, records are plain data)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    key: bytes | None
    value: bytes
    timestamp_ms: int | None = None


def process_record(record: Record, lookup: LookupFn) -> tuple[bytes, bytes]:
    """Boundary for one Kafka message -> (key, enriched bytes). A payload that is not a JSON
    object cannot carry an event, so it raises ValueError for the caller to log and skip."""
    try:
        event = json.loads(record.value)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"not valid JSON: {type(exc).__name__}") from exc
    if not isinstance(event, dict):
        raise ValueError("event is not a JSON object")
    msg_time = (datetime.fromtimestamp(record.timestamp_ms / 1000, tz=timezone.utc)
                if record.timestamp_ms else None)
    enriched = enrich_event(event, lookup, message_time=msg_time)
    key = record.key if record.key is not None else hashlib.sha1(record.value).hexdigest().encode()
    return key, json.dumps(enriched, ensure_ascii=False, separators=(",", ":")).encode()


def run(
    records: Iterable[Record],
    producer: Any,
    commit: Callable[[Record], None],
    lookup: LookupFn,
    output_topic: str,
    publish_timeout: float = 10.0,
) -> dict[str, int]:
    """Enrich every record, publish, and only then commit. A publish failure raises and leaves
    the offset uncommitted, so the message is redelivered and (being idempotent) re-enriched."""
    stats = {"published": 0, "skipped_malformed": 0}
    for record in records:
        try:
            key, payload = process_record(record, lookup)
        except ValueError as exc:
            logger.warning("skipping malformed message: %s", exc)
            stats["skipped_malformed"] += 1
            commit(record)
            continue
        producer.send(output_topic, key=key, value=payload).get(timeout=publish_timeout)
        commit(record)
        stats["published"] += 1
    return stats


class _Done:
    def get(self, timeout: float | None = None) -> None:
        return None


class MemoryProducer:
    """In-memory broker half for dry-run and tests."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes | None, bytes]] = []

    def send(self, topic: str, key: bytes | None = None, value: bytes = b"") -> _Done:
        self.sent.append((topic, key, value))
        return _Done()


# ----------------------------------------------------------------------------
# Live Kafka (lazy import; kafka-python)
# ----------------------------------------------------------------------------
def _live_records(cfg: Config) -> tuple[Iterator[Record], Any, Callable[[Record], None]]:
    try:
        from kafka import KafkaConsumer, KafkaProducer
        from kafka.errors import NoBrokersAvailable
    except ImportError as exc:
        raise RuntimeError("live mode needs `kafka-python` (not installed); use --dry-run offline") from exc

    def connect(factory: Callable[[], Any]) -> Any:
        for attempt in range(cfg.connect_retries + 1):
            try:
                return factory()
            except NoBrokersAvailable:
                logger.warning("kafka not reachable (attempt %d/%d)", attempt + 1, cfg.connect_retries + 1)
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"kafka at {cfg.bootstrap} unreachable after {cfg.connect_retries + 1} attempts")

    consumer = connect(lambda: KafkaConsumer(
        cfg.input_topic, bootstrap_servers=cfg.bootstrap, group_id=cfg.group_id,
        auto_offset_reset=cfg.auto_offset, enable_auto_commit=False,
    ))
    producer = connect(lambda: KafkaProducer(bootstrap_servers=cfg.bootstrap, acks="all"))

    def records() -> Iterator[Record]:
        for msg in consumer:
            yield Record(msg.key, msg.value, msg.timestamp)

    return records(), producer, lambda _r: consumer.commit()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _dry_run_records(path: Path | None) -> list[Record]:
    if path is None:
        lines = [json.dumps(e) for e in DRY_RUN_EVENTS]
    else:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [Record(None, ln.encode()) for ln in lines]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.services.enrich",
        description="Enrich pa5220.raw events against OpenCTI and publish to pa5220.enriched.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="stubbed OpenCTI and in-memory broker; no network, no .env required")
    parser.add_argument("--input", type=Path, help="dry-run only: JSONL file of raw events")
    parser.add_argument("--stub-fail", action="store_true", help="dry-run only: make the stub OpenCTI fail")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s [%(name)s] %(levelname)s %(message)s", level=logging.INFO)

    if args.dry_run:
        lookup = CachedLookup(StubOpenCTIClient(fail=args.stub_fail), max_size=1000)
        producer = MemoryProducer()
        stats = run(_dry_run_records(args.input), producer, lambda _r: None, lookup, "pa5220.enriched")
        for _topic, _key, value in producer.sent:
            print(value.decode())
        logger.info("dry-run done: %s, opencti queries=%d, cache entries=%d", stats, lookup.queries, len(lookup))
        return 0

    try:
        cfg = load_config()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    logger.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    lookup = CachedLookup(HttpOpenCTIClient(cfg.opencti_url, cfg.opencti_token), max_size=cfg.cache_size)
    try:
        records, producer, commit = _live_records(cfg)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    logger.info("enriching %s -> %s (group %s)", cfg.input_topic, cfg.output_topic, cfg.group_id)
    try:
        run(records, producer, commit, lookup, cfg.output_topic)
    except KeyboardInterrupt:
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
