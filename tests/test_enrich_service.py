"""Offline behavior tests for src/services/enrich.py. No broker, no OpenCTI, no network."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services import enrich
from src.services.enrich import (
    CachedLookup, Indicator, Lookup, MemoryProducer, OpenCTIError, Record, Related,
    StubOpenCTIClient, enrich_event, run,
)

HIT = "45.33.32.156"
MISS = "93.184.216.34"

RAW = {
    "timestamp": "2026-07-01T10:00:00Z",
    "src_ip": HIT,
    "dst_ip": "10.0.0.5",
    "dst_port": 22,
    "action": "allow",
    "nested": {"a": [1, 2]},
}


def fresh_lookup(**kw):
    stub = StubOpenCTIClient(**kw)
    return CachedLookup(stub, max_size=100), stub


def strip_source(event):
    out = copy.deepcopy(event)
    out["enrichment"].pop("source")
    return out


class EnrichEventTests(unittest.TestCase):
    def test_additive_and_never_drops_or_mutates_raw_fields(self):
        raw = copy.deepcopy(RAW)
        lookup, _ = fresh_lookup()
        out = enrich_event(raw, lookup)
        self.assertEqual(raw, RAW, "input must not be mutated")
        for key, value in RAW.items():
            self.assertEqual(out[key], value)
        self.assertEqual(set(out) - set(RAW), {"enrichment"})

    def test_match_carries_indicators_and_graph_walk(self):
        lookup, _ = fresh_lookup()
        e = enrich_event(RAW, lookup)["enrichment"]
        self.assertTrue(e["matched"])
        self.assertEqual(e["observables_checked"], [HIT])
        self.assertEqual(e["indicators"][0]["id"], "indicator--stub-1")
        self.assertEqual({r["type"] for r in e["related"]}, {"Intrusion-Set", "Attack-Pattern"})
        self.assertEqual(e["source"], "opencti")
        self.assertNotIn("error", e)

    def test_unmatched_event_still_carries_enrichment_object(self):
        lookup, _ = fresh_lookup()
        e = enrich_event({"timestamp": "2026-07-01T10:00:00Z", "src_ip": MISS}, lookup)["enrichment"]
        self.assertFalse(e["matched"])
        self.assertEqual(e["indicators"], [])
        self.assertEqual(e["related"], [])
        self.assertEqual(e["observables_checked"], [MISS])
        self.assertIn("enriched_at", e)

    def test_event_with_no_observables_still_has_enrichment(self):
        lookup, stub = fresh_lookup()
        out = enrich_event({"timestamp": "2026-07-01T10:00:00Z", "src_ip": "10.1.1.1"}, lookup)
        self.assertFalse(out["enrichment"]["matched"])
        self.assertEqual(out["enrichment"]["observables_checked"], [])
        self.assertEqual(stub.calls, [])

    def test_opencti_failure_publishes_with_error_string(self):
        lookup, _ = fresh_lookup(fail=True)
        out = enrich_event(RAW, lookup)
        e = out["enrichment"]
        self.assertFalse(e["matched"])
        self.assertIsInstance(e["error"], str)
        self.assertTrue(e["error"])
        self.assertEqual(out["src_ip"], HIT)

    def test_timestamp_is_added_when_missing_and_flagged_when_not_iso(self):
        lookup, _ = fresh_lookup()
        missing = enrich_event({"src_ip": MISS}, lookup)
        self.assertIsNotNone(enrich.parse_iso8601(missing["timestamp"]))
        bad = enrich_event({"timestamp": "2026/07/01 10:00:00", "src_ip": MISS}, lookup)
        self.assertEqual(bad["timestamp"], "2026/07/01 10:00:00", "raw value must not be rewritten")
        self.assertIn("timestamp_warning", bad["enrichment"])

    def test_cidr_indicator_matches_contained_ip_only(self):
        ind = Indicator("i1", "range", "[ipv4-addr:value = '45.33.32.0/24']", None, None)
        self.assertEqual(enrich.match_indicators(HIT, [ind]), (ind,))
        self.assertEqual(enrich.match_indicators("93.184.216.34", [ind]), ())

    def test_fuzzy_search_hit_for_a_different_ip_is_rejected(self):
        ind = Indicator("i1", "near", "[ipv4-addr:value = '45.33.32.157']", None, None)
        self.assertEqual(enrich.match_indicators(HIT, [ind]), ())


class IdempotencyTests(unittest.TestCase):
    def test_redelivery_yields_identical_output_with_cold_cache(self):
        rec = Record(b"k", json.dumps(RAW).encode(), timestamp_ms=1_782_900_000_000)
        a = enrich.process_record(rec, fresh_lookup()[0])
        b = enrich.process_record(rec, fresh_lookup()[0])
        self.assertEqual(a, b)

    def test_redelivery_through_warm_cache_differs_only_in_source(self):
        rec = Record(b"k", json.dumps(RAW).encode(), timestamp_ms=1_782_900_000_000)
        lookup, _ = fresh_lookup()
        first = json.loads(enrich.process_record(rec, lookup)[1])
        second = json.loads(enrich.process_record(rec, lookup)[1])
        self.assertEqual(first["enrichment"]["source"], "opencti")
        self.assertEqual(second["enrichment"]["source"], "cache")
        self.assertEqual(strip_source(first), strip_source(second))

    def test_enriched_at_does_not_depend_on_wall_clock(self):
        lookup, _ = fresh_lookup()
        a = enrich_event(RAW, lookup)["enrichment"]["enriched_at"]
        b = enrich_event(RAW, lookup)["enrichment"]["enriched_at"]
        self.assertEqual(a, b)


class CacheTests(unittest.TestCase):
    def test_repeated_ip_is_not_requeried(self):
        lookup, stub = fresh_lookup()
        for _ in range(5):
            enrich_event(RAW, lookup)
        self.assertEqual(stub.calls, [HIT])
        self.assertEqual(lookup.queries, 1)

    def test_negative_results_are_cached_too(self):
        lookup, stub = fresh_lookup()
        for _ in range(3):
            enrich_event({"src_ip": MISS}, lookup)
        self.assertEqual(stub.calls, [MISS])

    def test_cache_is_bounded_lru(self):
        stub = StubOpenCTIClient()
        lookup = CachedLookup(stub, max_size=2)
        for ip in ("8.8.8.8", "8.8.4.4", "1.1.1.1"):
            lookup(ip)
        self.assertEqual(len(lookup), 2)
        lookup("8.8.8.8")
        self.assertEqual(stub.calls.count("8.8.8.8"), 2, "oldest entry was evicted and re-queried")

    def test_ttl_expiry_requeries(self):
        t = [0.0]
        stub = StubOpenCTIClient()
        lookup = CachedLookup(stub, max_size=10, ttl_seconds=60, clock=lambda: t[0])
        lookup(MISS)
        t[0] = 61
        lookup(MISS)
        self.assertEqual(len(stub.calls), 2)

    def test_failures_are_not_cached_and_recovery_is_immediate(self):
        stub = StubOpenCTIClient(fail=True)
        lookup = CachedLookup(stub, max_size=10, breaker_threshold=99)
        with self.assertRaises(OpenCTIError):
            lookup(HIT)
        stub.fail = False
        result, source = lookup(HIT)
        self.assertEqual(source, "opencti")
        self.assertTrue(result.indicators)

    def test_circuit_breaker_stops_querying_then_recloses(self):
        t = [0.0]
        stub = StubOpenCTIClient(fail=True)
        lookup = CachedLookup(stub, max_size=10, breaker_threshold=2, breaker_cooldown=30, clock=lambda: t[0])
        for _ in range(2):
            with self.assertRaises(OpenCTIError):
                lookup(HIT)
        with self.assertRaises(OpenCTIError):
            lookup(HIT)
        self.assertEqual(len(stub.calls), 2, "third call short-circuited")
        t[0] = 31
        stub.fail = False
        self.assertEqual(lookup(HIT)[1], "opencti")


class RunLoopTests(unittest.TestCase):
    def records(self):
        return [Record(None, json.dumps(e).encode(), 1_782_900_000_000) for e in enrich.DRY_RUN_EVENTS]

    def test_publishes_every_event_and_commits_after_publish(self):
        order = []

        class Producer(MemoryProducer):
            def send(self, topic, key=None, value=b""):
                order.append("send")
                return super().send(topic, key, value)

        producer = Producer()
        lookup, _ = fresh_lookup()
        stats = run(self.records(), producer, lambda r: order.append("commit"), lookup, "pa5220.enriched")
        self.assertEqual(stats["published"], 3)
        self.assertEqual(order, ["send", "commit"] * 3)
        self.assertTrue(all(t == "pa5220.enriched" for t, _, _ in producer.sent))
        for _, _, value in producer.sent:
            ev = json.loads(value)
            self.assertIn("enrichment", ev)
            self.assertIsNotNone(enrich.parse_iso8601(ev["timestamp"]))

    def test_publish_failure_leaves_offset_uncommitted(self):
        committed = []

        class Boom:
            def send(self, *a, **k):
                raise RuntimeError("broker down")

        with self.assertRaises(RuntimeError):
            run(self.records(), Boom(), committed.append, fresh_lookup()[0], "pa5220.enriched")
        self.assertEqual(committed, [])

    def test_opencti_down_does_not_stop_the_pipeline(self):
        producer = MemoryProducer()
        lookup, _ = fresh_lookup(fail=True)
        stats = run(self.records(), producer, lambda r: None, lookup, "pa5220.enriched")
        self.assertEqual(stats["published"], 3)
        errors = [json.loads(v)["enrichment"].get("error") for _, _, v in producer.sent]
        self.assertTrue(errors[0] and errors[1])
        self.assertIsNone(errors[2], "internal-only event had nothing to look up, so no error")

    def test_malformed_message_is_skipped_and_committed(self):
        committed = []
        recs = [Record(None, b"not json"), Record(None, b"[1,2]")] + self.records()
        stats = run(recs, MemoryProducer(), committed.append, fresh_lookup()[0], "pa5220.enriched")
        self.assertEqual(stats, {"published": 3, "skipped_malformed": 2})
        self.assertEqual(len(committed), 5)


class BoundaryTests(unittest.TestCase):
    def test_parse_indicators_skips_malformed_nodes(self):
        payload = {"data": {"indicators": {"edges": [
            {"node": {"id": "i1", "name": "n", "pattern": "p", "confidence": 70, "valid_until": "2027-01-01T00:00:00Z"}},
            {"node": {"id": 5, "pattern": "p"}},
            {"node": {"id": "i2", "pattern": "p", "confidence": True}},
            None,
        ]}}}
        got = enrich.parse_indicators(payload)
        self.assertEqual([i.id for i in got], ["i1", "i2"])
        self.assertEqual(got[0].confidence, 70)
        self.assertIsNone(got[1].confidence)

    def test_parse_related_keeps_only_walk_target_types(self):
        payload = {"data": {"stixCoreRelationships": {"edges": [
            {"node": {"relationship_type": "indicates", "to": {"id": "m1", "entity_type": "Malware", "name": "X"}}},
            {"node": {"relationship_type": "related-to", "to": {"id": "c1", "entity_type": "Course-Of-Action", "name": "Y"}}},
        ]}}}
        self.assertEqual(enrich.parse_related(payload), [Related("m1", "Malware", "X", "indicates")])

    def test_extract_observables_filters_private_invalid_and_dedupes(self):
        ev = {"src_ip": "10.0.0.1", "dst_ip": HIT, "source_ip": HIT, "dst": "not-an-ip", "src": 5}
        self.assertEqual(enrich.extract_observables(ev), [HIT])

    def test_every_source_field_spelling_has_a_destination_counterpart(self):
        """IP_FIELDS was asymmetric: dest_ip and `Destination Address` were absent while
        destination_ip and `Destination address` were present, so a live event whose
        destination was spelled either way had that address silently never looked up.
        Lookup is keyed on the exact field name, so the asymmetry alone is the bug.
        """
        for field in ("dst_ip", "dest_ip", "destination_ip", "dst",
                      "Destination Address", "Destination address", "Destination IP"):
            with self.subTest(field=field):
                self.assertIn(field, enrich.IP_FIELDS)
                self.assertEqual(enrich.extract_observables({field: HIT}), [HIT])

    def test_walk_query_passes_fromid_as_a_list(self):
        """Live OpenCTI 5.12.17 types stixCoreRelationships(fromId:) as [String].

        Declaring $id as String! is accepted by no server and rejected at validation with
        GRAPHQL_VALIDATION_FAILED, which enrich.py reports as an OpenCTI failure -- so
        every graph walk returned no context while the service looked healthy. This is
        asserted against the query text because a unit test has no live platform; the
        shape itself was confirmed by running it against the real instance.
        """
        self.assertIn("$id: [String]", enrich._WALK_QUERY)
        self.assertNotIn("$id: String", enrich._WALK_QUERY)

    def test_config_fails_fast_naming_missing_keys_without_values(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as ctx:
                enrich.load_config(Path(d) / "absent.env", environ={"OPENCTI_TOKEN": "s3cret-value"})
        self.assertIn("KAFKA_BOOTSTRAP", str(ctx.exception))
        self.assertNotIn("s3cret-value", str(ctx.exception))

    def test_config_loads_and_hides_token_in_repr(self):
        env = {
            "KAFKA_BOOTSTRAP": "h:9092", "KAFKA_INPUT_TOPIC": "pa5220.raw", "KAFKA_OUTPUT_TOPIC": "pa5220.enriched",
            "KAFKA_GROUP_ID": "g", "KAFKA_AUTO_OFFSET": "earliest", "KAFKA_CONNECT_RETRIES": "3",
            "OPENCTI_URL": "http://x:8082/", "OPENCTI_TOKEN": "s3cret-value", "OPENCTI_CACHE_SIZE": "10",
        }
        with tempfile.TemporaryDirectory() as d:
            cfg = enrich.load_config(Path(d) / "absent.env", environ=env)
        self.assertEqual((cfg.cache_size, cfg.connect_retries, cfg.opencti_url), (10, 3, "http://x:8082"))
        self.assertNotIn("s3cret-value", repr(cfg))

    def test_dry_run_cli_runs_end_to_end(self):
        self.assertEqual(enrich.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()


class MetricsSinkTests(unittest.TestCase):
    """Telemetry must be optional and must never be able to stop enrichment."""

    def _cfg(self, **over):
        base = dict(bootstrap="b", input_topic="i", output_topic="o", group_id="g",
                    auto_offset="earliest", connect_retries=0,
                    opencti_url="http://x", opencti_token="t")
        base.update(over)
        return enrich.Config(**base)

    def test_disabled_without_full_influx_config(self):
        """Three of four keys is not a configured sink; it is a half-configured one."""
        self.assertFalse(enrich.MetricsSink(self._cfg()).enabled)
        partial = self._cfg(influx_url="http://i:8086", influx_token="tok", influx_org="soc")
        self.assertFalse(enrich.MetricsSink(partial).enabled,
                         "a missing bucket must disable the sink, not write to a default one")

    def test_disabled_sink_records_nothing_and_never_calls_out(self):
        sink = enrich.MetricsSink(self._cfg())
        with patch.object(enrich.urllib.request, "urlopen") as urlopen:
            sink.record_event({"matched": True, "source": "opencti"}, 1.5)
            sink.flush()
            sink.close()
        urlopen.assert_not_called()

    def _enabled(self, **over):
        return enrich.MetricsSink(self._cfg(
            influx_url="http://influx:8086", influx_token="tok",
            influx_org="soc", influx_bucket="alerts"), **over)

    def test_a_failing_write_does_not_propagate(self):
        """Losing telemetry is acceptable; losing enrichment because telemetry is down is not."""
        sink = self._enabled(batch_size=1)
        with patch.object(enrich.urllib.request, "urlopen", side_effect=OSError("influx down")):
            sink.record_event({"matched": False, "source": "opencti"}, 2.0)
            sink.close()  # must not raise

    def test_an_opencti_failure_is_tagged_error_not_a_clean_no_match(self):
        """An error still publishes with matched false, so without its own tag it would be
        counted as a genuine no-match and silently depress the IoC match rate."""
        sink = self._enabled(batch_size=1)
        sent = {}
        with patch.object(enrich.urllib.request, "urlopen") as urlopen:
            urlopen.side_effect = lambda req, timeout=None: sent.update(body=req.data) or MagicMock()
            sink.record_event({"matched": False, "source": "opencti", "error": "boom"}, 3.0)
        line = sent["body"].decode()
        self.assertIn("source=error", line)
        self.assertIn("matched=false", line)

    def test_line_protocol_shape_and_escaping(self):
        sink = self._enabled(batch_size=1)
        sent = {}
        with patch.object(enrich.urllib.request, "urlopen") as urlopen:
            urlopen.side_effect = lambda req, timeout=None: sent.update(body=req.data) or MagicMock()
            sink.record_event({"matched": True, "source": "cache",
                               "observables_checked": ["1.1.1.1", "8.8.8.8"],
                               "indicators": [{"id": "x"}]}, 12.25)
        line = sent["body"].decode()
        self.assertTrue(line.startswith("soc_enrich,matched=true,source=cache "), line)
        self.assertIn("latency_ms=12.250", line)
        self.assertIn("observables=2i", line)
        self.assertIn("indicators=1i", line)
        # integers must carry the i suffix or InfluxDB stores them as floats and the field
        # type conflicts with any later integer write to the same field
        self.assertNotIn("observables=2,", line)

    def test_tag_escaping_covers_the_characters_line_protocol_breaks_on(self):
        self.assertEqual(enrich._escape_tag("a,b c=d"), r"a\,b\ c\=d")

    def test_batching_holds_lines_until_the_batch_is_full(self):
        sink = self._enabled(batch_size=3, flush_interval_s=3600)
        with patch.object(enrich.urllib.request, "urlopen") as urlopen:
            for _ in range(2):
                sink.record_event({"matched": True, "source": "cache"}, 1.0)
            urlopen.assert_not_called()  # one HTTP call per event would cost more than enrichment
            sink.record_event({"matched": True, "source": "cache"}, 1.0)
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(urlopen.call_args[0][0].data.decode().count("\n"), 2)

    def test_run_flushes_the_sink_even_when_publishing_raises(self):
        """A publish failure is meant to propagate; it must not strand buffered measurements."""
        sink = self._enabled(batch_size=1000)
        producer = MagicMock()
        producer.send.return_value.get.side_effect = RuntimeError("broker gone")
        record = enrich.Record(key=None, value=b'{"src_ip": "8.8.8.8"}', timestamp_ms=0)
        with patch.object(enrich.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(RuntimeError):
                enrich.run([record], producer, lambda _r: None,
                           lambda _o: (enrich.Lookup((), ()), "opencti"), "out", sink=sink)
            urlopen.assert_called_once()
