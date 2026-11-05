"""Offline tests for src/pipeline.py -- the end-to-end entry point.

These run under Anaconda (`python -m unittest discover -s tests -p "test_[!l]*.py"`) and need
no Ollama, no broker and nothing from the uv .venv. src.pipeline is importable here precisely
because every heavy or .venv-only import lives inside a function, so what is tested is the
part that is pure: row selection, 94-feature selection, the positional identifier recovery and
its Label guard, alert-context assembly, and the Ollama reachability message.

Anything needing the LLM or the notification layer is skipped with an explicit reason.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from src import pipeline

ROOT = Path(__file__).resolve().parents[1]
HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None


def _fake_matrix(labels, n_features=pipeline.EXPECTED_FEATURES):
    """A frame shaped like the unified feature matrix: features + reference columns."""
    data = {f"f{i}": [float(i)] * len(labels) for i in range(n_features)}
    data["Label"] = list(labels)
    data["technique_id"] = ["T1046"] * len(labels)
    data["technique_name"] = ["Network Service Discovery"] * len(labels)
    data["tactic"] = ["discovery"] * len(labels)
    data["mapping_confidence"] = ["high"] * len(labels)
    data["technique_id_encoded"] = [1] * len(labels)
    data["tactic_encoded"] = [1] * len(labels)
    data["mapping_confidence_encoded"] = [0.9] * len(labels)
    data["is_known_technique"] = [1] * len(labels)
    data["orig_row_index"] = list(range(len(labels)))
    return pd.DataFrame(data)


class TestSelectRows(unittest.TestCase):
    LABELS = ["BENIGN", "PortScan", "BENIGN", "DDoS", "PortScan", "BENIGN"]

    def test_explicit_rows_are_returned_as_given(self):
        self.assertEqual(pipeline.select_rows(self.LABELS, rows=[3, 1]), [3, 1])

    def test_row_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            pipeline.select_rows(self.LABELS, rows=[99])

    def test_label_selection_only_picks_that_label(self):
        picked = pipeline.select_rows(self.LABELS, label="PortScan", random_n=2)
        self.assertEqual(sorted(picked), [1, 4])
        self.assertTrue(all(self.LABELS[i] == "PortScan" for i in picked))

    def test_label_defaults_to_one_row(self):
        self.assertEqual(len(pipeline.select_rows(self.LABELS, label="BENIGN")), 1)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            pipeline.select_rows(self.LABELS, label="Heartbleed")

    def test_asking_for_more_rows_than_exist_raises(self):
        with self.assertRaises(ValueError):
            pipeline.select_rows(self.LABELS, label="DDoS", random_n=3)

    def test_random_is_deterministic_under_a_seed(self):
        a = pipeline.select_rows(self.LABELS, random_n=3, seed=42)
        b = pipeline.select_rows(self.LABELS, random_n=3, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 3)

    def test_no_selection_mode_raises(self):
        with self.assertRaises(ValueError):
            pipeline.select_rows(self.LABELS)


class TestFeatureColumns(unittest.TestCase):
    def test_exactly_94_features_and_no_reference_columns(self):
        cols = pipeline.feature_columns(_fake_matrix(["BENIGN", "DDoS"]))
        self.assertEqual(len(cols), 94)
        for excluded in pipeline.NON_FEATURE_COLS:
            self.assertNotIn(excluded, cols)

    def test_wrong_feature_count_raises_rather_than_scoring_silently(self):
        with self.assertRaises(AssertionError):
            pipeline.feature_columns(_fake_matrix(["BENIGN"], n_features=92))

    def test_an_extra_numeric_column_is_caught(self):
        frame = _fake_matrix(["BENIGN"])
        frame["unexpected_numeric"] = [1.0]
        with self.assertRaises(AssertionError):
            pipeline.feature_columns(frame)

    @unittest.skipUnless(pipeline.UNIFIED_TEST.exists(), "unified test parquet not present")
    def test_real_unified_test_split_has_94_features(self):
        """The same selection the scoring stage performs, on the real matrix's real dtypes."""
        import pyarrow.parquet as pq

        first_group = pq.ParquetFile(pipeline.UNIFIED_TEST).read_row_group(0).to_pandas()
        cols = pipeline.feature_columns(first_group)
        self.assertEqual(len(cols), pipeline.EXPECTED_FEATURES)
        self.assertNotIn("Label", cols)
        self.assertNotIn("is_known_technique", cols)


class TestIdentifierRecovery(unittest.TestCase):
    """The positional index into the re-derived split is only safe because Label is asserted."""

    @staticmethod
    def _raw(labels, ips=None):
        return pd.DataFrame({
            "Label": list(labels),
            "Source IP": ips or [f"10.0.0.{i}" for i in range(len(labels))],
            "Destination IP": [f"192.168.1.{i}" for i in range(len(labels))],
            "Destination Port": [80 + i for i in range(len(labels))],
            "Timestamp": [dt.datetime(2017, 7, 5, 1, 0, i) for i in range(len(labels))],
        })

    def test_aligned_split_returns_the_matching_identifier_rows(self):
        labels = ["BENIGN", "PortScan", "DDoS", "BENIGN"]
        picked = pipeline.recover_identifiers(self._raw(labels), [1, 3], labels)
        self.assertEqual(list(picked["Label"]), ["PortScan", "BENIGN"])
        self.assertEqual(list(picked["Source IP"]), ["10.0.0.1", "10.0.0.3"])

    def test_misaligned_labels_raise(self):
        labels = ["BENIGN", "PortScan", "DDoS", "BENIGN"]
        shuffled = self._raw(["PortScan", "BENIGN", "DDoS", "BENIGN"])
        with self.assertRaises(AssertionError):
            pipeline.recover_identifiers(shuffled, [0, 1], labels)

    def test_row_count_mismatch_raises(self):
        labels = ["BENIGN", "PortScan", "DDoS"]
        with self.assertRaises(AssertionError):
            pipeline.recover_identifiers(self._raw(labels[:2]), [0], labels)


class TestAlertContext(unittest.TestCase):
    IDENT = {
        "Source IP": "172.16.0.1",
        "Destination IP": "192.168.10.50",
        "Destination Port": 445,
        "Timestamp": dt.datetime(2017, 7, 7, 3, 30, 0),
        "Label": "PortScan",
    }

    def _ctx(self, **kw):
        args = dict(
            rf_predicted_label="PortScan", fused_score=0.91, fused_prediction=1,
            confidence=0.98, model_used="random_forest", score_latency_s=0.25,
        )
        args.update(kw)
        return pipeline.build_alert_context(self.IDENT, **args)

    def test_carries_every_field_downstream_layers_read(self):
        ctx = self._ctx()
        for key in ("source_ip", "dst_ip", "dst_port", "timestamp", "ingested_at", "detected_at",
                    "rf_predicted_label", "fused_score", "fused_prediction", "confidence",
                    "model_used"):
            self.assertIn(key, ctx)
        self.assertEqual(ctx["source_ip"], "172.16.0.1")
        self.assertEqual(ctx["dst_port"], 445)
        self.assertIsInstance(ctx["dst_port"], int)

    def test_timestamps_are_isoformat_parseable(self):
        ctx = self._ctx()
        start = dt.datetime.fromisoformat(ctx["ingested_at"])
        end = dt.datetime.fromisoformat(ctx["detected_at"])
        self.assertAlmostEqual((end - start).total_seconds(), 0.25, places=6)

    def test_model_used_is_one_of_the_two_whitelisted_values(self):
        from src.modeling import fusion_layer

        for value in fusion_layer.MODEL_USED_VALUES:
            self.assertIn(self._ctx(model_used=value)["model_used"],
                          fusion_layer.MODEL_USED_VALUES)

    def test_rag_query_falls_back_for_benign_predictions(self):
        self.assertEqual(pipeline.rag_query_for("PortScan"), "PortScan")
        self.assertEqual(pipeline.rag_query_for("BENIGN"), pipeline.GENERIC_ANOMALY_QUERY)


class TestOllamaStatus(unittest.TestCase):
    def test_unreachable_backend_reports_a_reason_instead_of_raising(self):
        # Port 1 is reserved and never listening, so this exercises the failure branch
        # without depending on whether Ollama happens to be up on this machine.
        up, detail = pipeline.ollama_status("http://127.0.0.1:1", timeout_s=2.0)
        self.assertFalse(up)
        self.assertTrue(detail)


class TestSummaryRendering(unittest.TestCase):
    PAYLOAD = {
        "fusion_strategy": "confidence_weighted",
        "n_features": 94,
        "score_latency_ms_per_row": 17.0,
        "alerts": [{
            "test_row_index": 7,
            "true_label": "PortScan",
            "alert_context": {
                "source_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "dst_port": 445,
                "timestamp": "2017-07-07T03:30:00", "fused_score": 0.91, "fused_prediction": 1,
                "model_used": "if_ae_fallback", "confidence": 0.3,
                "rf_predicted_label": "PortScan",
            },
            "scores": {"rf_attack_probability": 0.6, "rf_predicted_label": "PortScan",
                       "if_anomaly_score": 0.5, "ae_reconstruction_error": 0.02,
                       "if_norm": 0.44, "ae_norm": 0.2, "margin": 0.88},
            "shap": None, "rag_query": "PortScan", "rag_hits": [],
        }],
    }

    def test_renders_without_llm_or_dispatch_sections(self):
        text = pipeline.render_summary(self.PAYLOAD)
        self.assertIn("row 7", text)
        self.assertIn("if_ae_fallback", text)
        self.assertIn("skipped (--no-llm)", text)

    def test_dry_run_dispatch_states_that_nothing_was_sent(self):
        payload = {**self.PAYLOAD, "alerts": [dict(self.PAYLOAD["alerts"][0])]}
        payload["alerts"][0]["llm_output"] = {
            "priority": "HIGH", "explanation_th": "ทดสอบ", "technique_id": "T1046",
            "mitre_tactic": "discovery", "remediation": ["บล็อก IP"],
        }
        payload["alerts"][0]["dispatch"] = {
            "status": "dry_run", "would_route_to": "LINE (immediate)",
            "routing_key": "alert.HIGH",
            "record": {"risk_score": 91, "model_used": "if_ae_fallback",
                       "anomaly_type": "PortScan", "mttd_ms": 250.0},
        }
        text = pipeline.render_summary(payload)
        self.assertIn("DRY RUN -- nothing sent", text)
        self.assertIn("alert.HIGH", text)

    def test_renders_a_real_dispatch_receipt(self):
        """A real dispatch_alert receipt has no top-level "status".

        It carries alert_id and priority plus one status per sink. render_summary
        subscripted d["status"], so every --dispatch run raised KeyError *after* the alert
        had already been published and the LINE message sent -- the one shape the dry-run
        default never produces, so nothing caught it until the first live run. The receipt
        below is the exact shape dispatch_alert returned on that run.
        """
        payload = {**self.PAYLOAD, "alerts": [dict(self.PAYLOAD["alerts"][0])]}
        payload["alerts"][0]["llm_output"] = {
            "priority": "HIGH", "explanation_th": "ทดสอบ", "technique_id": "T1046",
            "mitre_tactic": "discovery", "remediation": ["บล็อก IP"],
        }
        payload["alerts"][0]["dispatch"] = {
            "alert_id": "bfff10e2-6b8e-446b-a930-1169b3841f47", "priority": "HIGH",
            "risk_score": 100, "model_used": "random_forest",
            "rabbitmq": {"status": "ok", "exchange": "soc.alerts", "routing_key": "alert.HIGH"},
            "influxdb": {"status": "ok"}, "elasticsearch": {"status": "ok"},
            "notification": {"status": "sent", "delivery_mode": "broadcast", "http_status": 200},
        }
        text = pipeline.render_summary(payload)
        self.assertIn("bfff10e2-6b8e-446b-a930-1169b3841f47", text)
        self.assertIn("notification=sent", text)
        self.assertNotIn("DRY RUN", text)


class TestCliDefaults(unittest.TestCase):
    def test_dry_run_is_the_default_and_dispatch_is_opt_in(self):
        args = pipeline.build_parser().parse_args(["--random", "2"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.dispatch)

    def test_scoring_subprocess_gets_the_selection_flags(self):
        args = pipeline.build_parser().parse_args(["--label", "PortScan", "--random", "2", "--shap"])
        forwarded = pipeline._forward_args(args)
        self.assertEqual(forwarded[:2], ["--stage", "score"])
        self.assertIn("--quiet", forwarded)
        self.assertIn("--shap", forwarded)
        self.assertIn("PortScan", forwarded)


class TestStagesNeedingTheOtherInterpreter(unittest.TestCase):
    """Explicit skips, never a silent pass: these two stages cannot run under Anaconda."""

    @unittest.skipUnless(HAS_LANGCHAIN,
                         "llm_stage imports src.llm.llm_explain -> langchain, which only the uv "
                         ".venv has; exercise it with `uv run python -m src.pipeline`")
    def test_llm_stage_is_importable(self):
        pipeline.llm_stage  # noqa: B018 -- presence only; a live run needs Ollama too

    @unittest.skip("dispatch_stage imports alert_dispatcher -> pika/elasticsearch/influxdb_client, "
                   "which only the uv .venv has; covered there by tests/test_alert_dispatcher.py")
    def test_dispatch_stage_against_a_broker(self):
        raise AssertionError("not runnable here")


if __name__ == "__main__":
    unittest.main()
