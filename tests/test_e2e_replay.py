"""Offline tests for src/e2e_replay.py -- the replay-based end-to-end runner.

These run under Anaconda (`python -m unittest discover -s tests -p "test_[!l]*.py"`) with no
broker, no models, no Ollama and no parquet. src.e2e_replay imports kafka, torch and pandas
only inside the stage functions, so what is exercised here is the part that decides whether
the chain is trustworthy: the event payload, the stage registry and the coordinator that
walks it, the consume stage's refusal to pass a partial set, and the precondition checks.
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from src import e2e_replay as E
from src import pipeline

FEATURES = [f"feat_{i}" for i in range(pipeline.EXPECTED_FEATURES)]
IDENT = {
    "Timestamp": dt.datetime(2017, 7, 5, 10, 47, 3),
    "Source IP": "205.174.165.73", "Destination IP": "192.168.10.50",
    "Destination Port": 80.0, "Protocol": 6.0,
}


def make_event(n=0, run_id="r1", values=None):
    values = values if values is not None else [float(i) / 7 for i in range(len(FEATURES))]
    return E.build_event(run_id, n, IDENT, 44999, "PortScan", FEATURES, values)


class TestEventPayload(unittest.TestCase):
    def test_round_trip_keeps_all_94_feature_names_and_exact_values(self):
        event = make_event()
        back = json.loads(E.dumps_event(event))
        self.assertEqual(back, event)
        self.assertEqual(list(back["features"]), FEATURES)
        self.assertEqual(len(back["features"]), 94)

    def test_identifiers_use_the_spellings_enrich_looks_up(self):
        from src.services import enrich

        event = make_event()
        self.assertIn("src_ip", enrich.IP_FIELDS)
        self.assertIn("dest_ip", enrich.IP_FIELDS)
        self.assertEqual(event["src_ip"], "205.174.165.73")
        self.assertEqual(event["dest_ip"], "192.168.10.50")
        self.assertEqual(event["dest_port"], 80)
        self.assertEqual(event["protocol"], 6)
        self.assertEqual(event["event_id"], "e2e-r1-0")
        self.assertEqual(event["timestamp"], "2017-07-05T10:47:03")

    def test_nan_is_rejected_at_serialization(self):
        values = [0.0] * len(FEATURES)
        values[3] = float("nan")
        with self.assertRaises(ValueError):
            E.dumps_event(make_event(values=values))

    def test_mismatched_names_and_values_are_rejected(self):
        with self.assertRaises(ValueError):
            E.build_event("r", 0, IDENT, 1, "x", FEATURES, [0.0])

    def test_jsonl_file_round_trip(self):
        events = [make_event(0), make_event(1)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "events.jsonl"
            self.assertEqual(E.write_jsonl(path, events), 2)
            self.assertEqual(E.load_jsonl(path), events)


class TestVectorize(unittest.TestCase):
    def test_selects_by_name_not_by_dict_order(self):
        features = {name: float(i) for i, name in enumerate(FEATURES)}
        shuffled = dict(reversed(list(features.items())))
        self.assertEqual(E.vectorize(shuffled, FEATURES), E.vectorize(features, FEATURES))
        self.assertEqual(E.vectorize(shuffled, FEATURES)[5], 5.0)

    def test_missing_or_unexpected_feature_is_an_error_not_a_silent_shift(self):
        features = {name: 0.0 for name in FEATURES}
        with self.assertRaises(ValueError):
            E.vectorize({k: v for k, v in features.items() if k != "feat_7"}, FEATURES)
        with self.assertRaises(ValueError):
            E.vectorize({**features, "surprise": 1.0}, FEATURES)


class TestAdditiveEnrichment(unittest.TestCase):
    RAW = {"event_id": "a", "src_ip": "1.2.3.4", "features": {"x": 1.0}}

    def test_pure_addition_has_no_problems(self):
        enriched = {**self.RAW, "enrichment": {"matched": False}}
        self.assertEqual(E.verify_additive(self.RAW, enriched), [])

    def test_changed_dropped_and_extra_fields_are_reported(self):
        changed = {**self.RAW, "src_ip": "9.9.9.9", "enrichment": {}}
        dropped = {"event_id": "a", "src_ip": "1.2.3.4", "enrichment": {}}
        extra = {**self.RAW, "enrichment": {}, "score": 0.5}
        self.assertTrue(any("src_ip" in p for p in E.verify_additive(self.RAW, changed)))
        self.assertTrue(any("features" in p for p in E.verify_additive(self.RAW, dropped)))
        self.assertTrue(any("score" in p for p in E.verify_additive(self.RAW, extra)))
        self.assertTrue(E.verify_additive(self.RAW, self.RAW))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def message(event_id, **extra):
    return json.dumps({"event_id": event_id, "enrichment": {}, **extra}).encode()


class TestConsume(unittest.TestCase):
    def test_collects_all_expected_events_and_ignores_the_rest(self):
        clock = FakeClock()
        batches = iter([[message("other"), b"not json"], [message("a"), message("a"), message("b")]])
        found, missing = E.collect_enriched(["a", "b"], lambda t: next(batches, []), 30, clock)
        self.assertEqual(sorted(found), ["a", "b"])
        self.assertEqual(missing, [])

    def test_missing_ids_are_reported_when_the_deadline_passes(self):
        clock = FakeClock()

        def poll(timeout_s):
            clock.now += timeout_s
            return [message("a")]

        found, missing = E.collect_enriched(["a", "b", "c"], poll, 5, clock)
        self.assertEqual(list(found), ["a"])
        self.assertEqual(missing, ["b", "c"])

    def test_stage_fails_naming_the_missing_ids_and_how_to_start_enrich(self):
        raw = {"a": {"event_id": "a"}, "b": {"event_id": "b"}}
        clock = FakeClock()

        def poll(timeout_s):
            clock.now += timeout_s
            return [message("a")]

        with self.assertRaises(E.StageError) as ctx:
            E.resolve_enriched(["a", "b"], raw, poll, 3, "pa5220.enriched", clock)
        text = str(ctx.exception)
        self.assertIn("'b'", text)
        self.assertNotIn("'a'", text)
        self.assertIn("src.services.enrich", text)
        self.assertIn("--from consume", text)

    def test_stage_returns_events_in_expected_order_when_complete(self):
        raw = {"a": {"event_id": "a"}, "b": {"event_id": "b"}}
        batches = iter([[message("b"), message("a")]])
        out = E.resolve_enriched(["a", "b"], raw, lambda t: next(batches, []), 5, "t")
        self.assertEqual([e["event_id"] for e in out], ["a", "b"])

    def test_stage_rejects_an_enrichment_that_altered_the_raw_event(self):
        raw = {"a": {"event_id": "a", "src_ip": "1.2.3.4"}}
        batches = iter([[message("a", src_ip="6.6.6.6")]])
        with self.assertRaises(E.StageError) as ctx:
            E.resolve_enriched(["a"], raw, lambda t: next(batches, []), 5, "t")
        self.assertIn("src_ip", str(ctx.exception))


class TestRegistry(unittest.TestCase):
    def test_declared_order_and_interpreters(self):
        self.assertEqual([s.key for s in E.STAGES], ["export", "produce", "consume", "score", "notify"])
        by_key = {s.key: s.interpreter for s in E.STAGES}
        self.assertEqual(by_key["export"], E.ANACONDA)
        self.assertEqual(by_key["score"], E.ANACONDA)
        for key in ("produce", "consume", "notify"):
            self.assertEqual(by_key[key], E.VENV)

    def test_each_stage_consumes_what_the_previous_one_produced(self):
        self.assertIsNone(E.STAGES[0].needs)
        for prev, nxt in zip(E.STAGES, E.STAGES[1:]):
            self.assertEqual(nxt.needs, prev.produces, nxt.key)

    def test_from_starts_at_the_named_stage(self):
        self.assertEqual([s.key for s in E.plan(E.STAGES, "score")], ["score", "notify"])
        self.assertEqual([s.key for s in E.plan(E.STAGES, None)], [s.key for s in E.STAGES])
        with self.assertRaises(E.StageError):
            E.plan(E.STAGES, "nope")

    def test_dry_run_is_the_default_and_dispatch_opts_in(self):
        self.assertFalse(E.build_parser().parse_args([]).dispatch)
        self.assertTrue(E.build_parser().parse_args(["--dispatch"]).dispatch)
        self.assertEqual(E.build_parser().parse_args(["--from", "score"]).from_stage, "score")


class TestCoordinator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        d = Path(self._tmp.name)
        self.a, self.b, self.c = d / "a.out", d / "b.out", d / "c.out"
        self.stages = (
            E.Stage("one", "", E.VENV, None, self.a, lambda args: None),
            E.Stage("two", "", E.VENV, self.a, self.b, lambda args: None),
            E.Stage("three", "", E.VENV, self.b, self.c, lambda args: None),
        )
        self.ran: list[str] = []

    def executor(self, skip_write=(), fail=()):
        def execute(stage, args):
            self.ran.append(stage.key)
            if stage.key in fail:
                raise E.StageError("boom")
            if stage.key not in skip_write:
                stage.produces.write_text("data")
        return execute

    def test_walks_every_stage_in_order(self):
        results = E.run_chain(self.stages, None, None, self.executor())
        self.assertEqual(self.ran, ["one", "two", "three"])
        self.assertTrue(all(r.ok for r in results))

    def test_from_skips_earlier_stages_and_uses_their_artifacts(self):
        self.a.write_text("old")
        results = E.run_chain(self.stages, "two", None, self.executor())
        self.assertEqual(self.ran, ["two", "three"])
        self.assertEqual([r.key for r in results], ["two", "three"])
        self.assertEqual(self.a.read_text(), "old")

    def test_missing_input_artifact_stops_before_the_stage_runs(self):
        results = E.run_chain(self.stages, "two", None, self.executor())
        self.assertEqual(self.ran, [])
        self.assertFalse(results[-1].ok)
        self.assertIn("a.out", results[-1].detail)

    def test_a_stage_that_writes_nothing_fails_even_if_a_stale_artifact_exists(self):
        self.b.write_text("stale from an earlier run")
        results = E.run_chain(self.stages, None, None, self.executor(skip_write=("two",)))
        self.assertFalse(results[-1].ok)
        self.assertEqual(results[-1].key, "two")
        self.assertEqual(self.ran, ["one", "two"])

    def test_a_failing_stage_stops_the_chain_and_is_reported_last(self):
        results = E.run_chain(self.stages, None, None, self.executor(fail=("two",)))
        self.assertEqual(self.ran, ["one", "two"])
        self.assertEqual([(r.key, r.ok) for r in results], [("one", True), ("two", False)])
        self.assertIn("boom", results[-1].detail)

    def test_an_in_place_stage_keeps_its_input(self):
        inplace = (E.Stage("rw", "", E.VENV, self.a, self.a, lambda args: None),)
        self.a.write_text("payload")
        seen = []

        def execute(stage, args):
            seen.append(stage.produces.read_text())

        results = E.run_chain(inplace, None, None, execute)
        self.assertEqual(seen, ["payload"])
        self.assertTrue(results[-1].ok)


class TestChecks(unittest.TestCase):
    def test_a_raising_check_is_a_fail_with_the_reason_not_an_exception(self):
        def boom():
            raise ConnectionError("broker down")

        results = E.run_checks([E.Check("ok", lambda: "fine"), E.Check("kafka", boom)])
        self.assertEqual([(r.name, r.ok) for r in results], [("ok", True), ("kafka", False)])
        self.assertIn("broker down", results[1].detail)

    def test_missing_model_files_fail_and_name_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            result = E.run_checks([E.Check("models", lambda: E.check_models(Path(d)))])[0]
        self.assertFalse(result.ok)
        self.assertIn("random_forest_unified.pkl", result.detail)

    def test_all_model_files_present_passes(self):
        with tempfile.TemporaryDirectory() as d:
            for f in E.REQUIRED_MODEL_FILES:
                p = Path(d) / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            self.assertTrue(E.run_checks([E.Check("models", lambda: E.check_models(Path(d)))])[0].ok)

    def test_the_default_checks_cover_every_dependency_the_chain_needs(self):
        names = " ".join(c.name for c in E.default_checks())
        for needle in ("kafka", "enrich.py running", "model files", "ollama", "rabbitmq", "line"):
            self.assertIn(needle, names)


class TestSummary(unittest.TestCase):
    @staticmethod
    def alert(enrichment, dispatch, suppressed=False):
        return {
            "test_row_index": 7, "true_label": "PortScan", "event_id": "e2e-r1-0",
            "suppressed": suppressed, "enrichment": enrichment, "dispatch": dispatch,
            "llm_output": None if suppressed else {"priority": "HIGH"},
            "alert_context": {"model_used": "random_forest", "fused_score": 0.99,
                              "timestamp": "2017-07-05T10:47:03"},
        }

    def test_completion_requires_enrichment_and_a_dispatch_outcome(self):
        ok = {"rabbitmq": {"status": "ok"}, "influxdb": {"status": "ok"},
              "elasticsearch": {"status": "ok"}, "notification": {"status": "sent"}}
        bad = {**ok, "influxdb": {"status": "error"}}
        self.assertTrue(E.alert_completed(self.alert({"matched": False}, ok)))
        self.assertTrue(E.alert_completed(self.alert({"matched": False}, {"status": "dry_run"})))
        self.assertFalse(E.alert_completed(self.alert({"matched": False}, bad)))
        self.assertFalse(E.alert_completed(self.alert({"matched": False}, {"status": "skipped"})))
        self.assertFalse(E.alert_completed(self.alert(None, {"status": "dry_run"})))
        self.assertFalse(E.alert_completed(self.alert({"matched": False}, None)))

    def test_summary_reports_enrichment_dispatch_timing_and_replay_dates(self):
        ok = {"rabbitmq": {"status": "ok"}, "influxdb": {"status": "ok"},
              "elasticsearch": {"status": "ok"}, "notification": {"status": "sent"}}
        payload = {"dry_run": False, "alerts": [
            self.alert({"matched": True, "indicators": [{"id": "i"}]}, ok),
            self.alert({"matched": False, "observables_checked": ["8.8.8.8"]}, None),
        ]}
        text = E.render_e2e_summary(payload, [E.StageResult("export", 1.5, True)])
        self.assertIn("matched (1 indicator(s))", text)
        self.assertIn("no match (1 public IP(s) checked)", text)
        self.assertIn("rabbitmq=ok influxdb=ok elasticsearch=ok notification=sent", text)
        self.assertIn("1 of 2 event(s) completed the whole chain", text)
        self.assertIn("export", text)
        self.assertIn("soc-alerts-2017.07.05", text)


if __name__ == "__main__":
    unittest.main()
