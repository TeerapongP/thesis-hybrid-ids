# -*- coding: utf-8 -*-
"""Keep CI's explicit test list and this environment's real capabilities in agreement.

The suite is split across two interpreters: the uv `.venv` holds the service dependencies
(LangChain, pika, elasticsearch, influxdb) and the system interpreter holds the ML ones
(pandas, scikit-learn, shap, a CUDA torch pinned to this machine's GPU). CI installs only
the first, so it can only run the test files that import without pandas.

It used to run `discover -p "test_*.py"` instead, which loaded all 23 files, failed 11 on
import, and left CI red on every push -- a red that said nothing about the code. Listing
the runnable modules explicitly fixes that, and introduces a new way to be wrong: a test
file added later would simply not be in the list, and CI would stay green while never
running it. That is worse than red, because it looks like coverage.

So the list is checked against reality here rather than trusted. This test runs in CI
itself, under the same environment CI uses, which is the only place the check is
meaningful.
"""
from __future__ import annotations

import importlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TESTS_DIR = ROOT / "tests"


def modules_listed_in_ci() -> set[str]:
    """The `tests.<name>` modules the workflow passes to unittest."""
    text = WORKFLOW.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*tests\.(test_[a-z0-9_]+)\s*$", text, re.MULTILINE))


def is_ci_environment() -> bool:
    """True in the environment CI runs in, identified by the dependency it lacks.

    Keyed on pandas rather than on "some module failed to import": the ML interpreter also
    has failures -- it has no LangChain -- so a check for any failure fires in both
    environments and the comparison below is meaningless in one of them.
    """
    try:
        importlib.import_module("pandas")
    except ModuleNotFoundError:
        return True
    return False


def modules_importable_here() -> tuple[set[str], dict[str, str]]:
    """Which test modules this interpreter can import, and why the rest cannot."""
    importable: set[str] = set()
    blocked: dict[str, str] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        name = path.stem
        try:
            importlib.import_module(f"tests.{name}")
            importable.add(name)
        except Exception as exc:  # an import error is the signal, not a failure
            blocked[name] = f"{type(exc).__name__}: {exc}"
    return importable, blocked


class CiTestSplitTests(unittest.TestCase):
    def test_workflow_lists_modules_at_all(self):
        """A regression to `discover` would silently restore the all-red state."""
        self.assertTrue(WORKFLOW.exists(), f"missing {WORKFLOW}")
        listed = modules_listed_in_ci()
        self.assertTrue(listed, "the workflow names no tests.<module>; did it revert to discover?")
        self.assertNotIn('discover -s tests -p "test_*.py"', WORKFLOW.read_text(encoding="utf-8"),
                         "CI discovers the whole suite again, which cannot pass without pandas")

    def test_ci_list_matches_what_this_environment_can_import(self):
        if not is_ci_environment():
            self.skipTest("pandas is present, so this is the ML interpreter and not the "
                          "environment CI runs in; the comparison would be meaningless")
        importable, _ = modules_importable_here()
        listed = modules_listed_in_ci()
        missing = sorted(importable - listed)
        phantom = sorted(listed - importable)

        self.assertFalse(
            missing,
            "these test modules import fine here but CI does not run them, so CI is green "
            f"while skipping them -- add them to .github/workflows/ci.yml: {missing}",
        )
        self.assertFalse(
            phantom,
            "CI lists these modules but they cannot be imported in this environment, so the "
            f"job will fail on load: {phantom}",
        )

    def test_blocked_modules_are_blocked_only_by_the_ml_dependencies(self):
        """A file excluded for the documented reason is fine; one excluded by a real bug is not.

        Without this, a genuine import error -- a syntax error, a bad relative import, a
        missing service dependency -- would look identical to "needs pandas" and be excluded
        from CI by the test above rather than reported.
        """
        if not is_ci_environment():
            self.skipTest("ML interpreter; what it blocks is LangChain, not the ML stack")
        _, blocked = modules_importable_here()
        expected = ("pandas", "shap", "torch", "sklearn", "imblearn", "joblib", "matplotlib")
        for name, error in sorted(blocked.items()):
            with self.subTest(module=name):
                self.assertTrue(
                    error.startswith("ModuleNotFoundError") and any(d in error for d in expected),
                    f"{name} is excluded from CI but not because of an ML dependency: {error}",
                )


if __name__ == "__main__":
    unittest.main()
