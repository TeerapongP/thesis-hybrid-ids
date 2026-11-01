"""The `model_used` strings must agree across the modules that produce and consume them.

This has broken before. `fusion_layer.fuse_with_metadata` emits a label naming which member
decided each row, and `alert_dispatcher.derive_model_used` whitelists those labels; anything
outside the whitelist becomes "unknown". When the fusion branch was renamed without migrating
the consumer, every alert decided by the renamed branch was silently relabelled -- no error,
no log line that looked wrong, just a wrong field on every affected alert.

The two lists are duplicated on purpose: fusion_layer pulls in numpy, while the notification
service runs on the lean .venv and must not. This test is what makes the duplication safe.

It reads alert_dispatcher's whitelist out of the SOURCE TEXT rather than importing it,
because importing it needs pika/elasticsearch/influxdb_client (.venv only) while importing
fusion_layer needs numpy (Anaconda only) -- no single interpreter can import both, so a test
that tried would fail for reasons unrelated to the contract it is checking.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUSION_LAYER = ROOT / "src/modeling/fusion_layer.py"
DISPATCHER = ROOT / "src/notifications/alert_dispatcher.py"


def _literal_tuple(source: str, pattern: str, label: str) -> tuple[str, ...]:
    match = re.search(pattern, source, re.MULTILINE)
    if match is None:
        raise AssertionError(
            f"could not locate the {label} literal. If it was refactored, update this test "
            f"rather than deleting it -- the contract it guards is still live."
        )
    value = ast.literal_eval(match.group(1))
    return tuple(value)


class ModelUsedContractTest(unittest.TestCase):
    def test_declared_values_match_the_dispatcher_whitelist(self):
        declared = _literal_tuple(
            FUSION_LAYER.read_text(encoding="utf-8"),
            r"^MODEL_USED_VALUES\s*=\s*(\([^)]*\))",
            "fusion_layer.MODEL_USED_VALUES",
        )
        whitelisted = _literal_tuple(
            DISPATCHER.read_text(encoding="utf-8"),
            r"if model_used not in (\([^)]*\)):",
            "alert_dispatcher.derive_model_used whitelist",
        )
        self.assertEqual(
            set(declared), set(whitelisted),
            "fusion_layer emits model_used values the dispatcher does not accept (or vice "
            "versa). Every alert decided by an unaccepted branch is labelled 'unknown'.",
        )

    def test_fusion_layer_emits_only_declared_values(self):
        """The np.where branches in fuse_with_metadata must use the declared strings."""
        source = FUSION_LAYER.read_text(encoding="utf-8")
        declared = set(_literal_tuple(
            source, r"^MODEL_USED_VALUES\s*=\s*(\([^)]*\))", "fusion_layer.MODEL_USED_VALUES",
        ))
        body = source.split("def fuse_with_metadata", 1)[1]
        emitted = set(re.findall(r'model_used = np\.where\([^,]+,\s*"([^"]+)",\s*"([^"]+)"\)', body)[0]) \
            if re.search(r"model_used = np\.where", body) else set()
        self.assertTrue(emitted, "fuse_with_metadata no longer assigns model_used via np.where; "
                                 "update this test to match the new mechanism.")
        self.assertEqual(
            emitted, declared,
            f"fuse_with_metadata emits {sorted(emitted)} but MODEL_USED_VALUES declares "
            f"{sorted(declared)}.",
        )


if __name__ == "__main__":
    unittest.main()
