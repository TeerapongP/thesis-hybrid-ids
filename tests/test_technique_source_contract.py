"""The evaluator must measure the configuration the system actually runs.

`src/evaluation/eval_llm_faithfulness.py` cannot import `llm_explain` at module level: its
alert-building stage runs under Anaconda, which has no langchain. So it keeps its own copy of
the deployed technique source, and this test is what stops the copy drifting. Both sides are
read from source text, because `llm_explain` imports only in the uv `.venv` and the evaluator
only in Anaconda, and this file must load in both.

The same arrangement guards `fusion_layer.MODEL_USED_VALUES` against
`alert_dispatcher.derive_model_used`, for the same reason and after that pair did drift once.

The failure this prevents is silent: a stale copy makes every M2 figure describe a
configuration nobody deployed, and nothing raises.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "src" / "evaluation" / "eval_llm_faithfulness.py"
DEPLOYED = ROOT / "src" / "llm" / "llm_explain.py"


def literal(path: Path, pattern: str, label: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, (f"no {label} found in {path.name}; if it was refactored, update this test "
                   f"rather than deleting it")
    return match.group(1)


def evaluator_technique_source() -> str:
    return literal(EVALUATOR, r'^TECHNIQUE_SOURCE = "([a-z]+)"', "module-level TECHNIQUE_SOURCE")


def deployed_technique_source() -> str:
    return literal(DEPLOYED, r'^DEFAULT_TECHNIQUE_SOURCE[^=\n]*= "([a-z]+)"',
                   "DEFAULT_TECHNIQUE_SOURCE")


def accepted_technique_sources() -> tuple[str, ...]:
    body = literal(DEPLOYED, r"^TECHNIQUE_SOURCES[^=\n]*= \(([^)]*)\)", "TECHNIQUE_SOURCES")
    return tuple(re.findall(r'"([a-z]+)"', body))


def evaluator_retrieval_context() -> str:
    return literal(EVALUATOR, r'^RETRIEVAL_CONTEXT = "([a-z_]+)"', "module-level RETRIEVAL_CONTEXT")


def deployed_retrieval_context() -> str:
    return literal(DEPLOYED, r'^DEFAULT_RETRIEVAL_CONTEXT[^=\n]*= "([a-z_]+)"',
                   "DEFAULT_RETRIEVAL_CONTEXT")


def accepted_retrieval_contexts(path: Path, name: str) -> tuple[str, ...]:
    body = literal(path, rf"^{name}[^=\n]*= \(([^)]*)\)", name)
    return tuple(re.findall(r'"([a-z_]+)"', body))


class TechniqueSourceContract(unittest.TestCase):
    def test_evaluator_default_equals_the_deployed_default(self):
        self.assertEqual(evaluator_technique_source(), deployed_technique_source())

    def test_the_value_is_one_the_resolver_accepts(self):
        self.assertIn(evaluator_technique_source(), accepted_technique_sources())


class RetrievalContextContract(unittest.TestCase):
    """Same arrangement as the technique source: the evaluator keeps its own copy."""

    def test_evaluator_default_equals_the_deployed_default(self):
        self.assertEqual(evaluator_retrieval_context(), deployed_retrieval_context())

    def test_the_value_is_one_the_prompt_builder_accepts(self):
        self.assertIn(evaluator_retrieval_context(),
                      accepted_retrieval_contexts(DEPLOYED, "RETRIEVAL_CONTEXTS"))

    def test_both_sides_offer_the_same_choices(self):
        self.assertEqual(accepted_retrieval_contexts(EVALUATOR, "RETRIEVAL_CONTEXTS"),
                         accepted_retrieval_contexts(DEPLOYED, "RETRIEVAL_CONTEXTS"))


if __name__ == "__main__":
    unittest.main()
