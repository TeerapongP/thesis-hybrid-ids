"""
shap_explain.py -- SHAP explainability over the deployed `confidence_weighted` fusion layer.

ATTRIBUTION CONTRACT (decided, and why)
---------------------------------------
The deployed score is a cascade (src/modeling/fusion_layer.py):

    fused = rf_attack_probability        where |rf_prob - 0.5| * 2 >= 0.95
            (if_norm + ae_norm) / 2     otherwise

Each row is therefore decided by exactly one of two mechanisms, and `fuse_with_metadata`
names it in `model_used`. This module attributes the mechanism that actually decided the row,
and attributes it in the units of the fused score, so `base_value + sum(shap) == fused_score`
holds for every record. That additivity is checked, not asserted -- see `additivity_gap`.

  * "random_forest" rows -> `shap.TreeExplainer` on the RF, explaining
    `rf_attack_probability = 1 - P(BENIGN)`. SHAP values for class BENIGN are negated so the
    sum reconstructs the attack probability, which IS the fused score on these rows.

  * "if_ae_fallback" rows -> `shap.KernelExplainer` over the COMPOSITE function
    `(if_norm + ae_norm) / 2`, evaluated through the same deployed normalization constants.
    This is the quantity that set the score, so the attribution explains the decision rather
    than one input to it.

WHY THE COMPOSITE IS ATTRIBUTABLE AT ALL, which it was not before. This module previously
served the retired `rf_ae_or` rule, where RF read 80 columns and the AE read a separate
92-column behavioral matrix joined on `orig_row_index`. Averaging two scores across two
incompatible feature spaces has no meaningful attribution, and `max()` has no single input
space either. On the unified population all three members read the SAME 94 columns, so the
composite is an ordinary scalar function of one feature vector and KernelExplainer applies
directly. The data-layer fix is what made this explainable; it was not a code problem.

A property worth stating: on fallback rows the Isolation Forest genuinely influences the
attribution. Under the OR rule IF provably could not affect the decision, and this module
used to pass `np.zeros` in its place. Doing that now would change the fused score and
mislabel every fallback row, so it is gone.

BACKGROUND SET. 100 BENIGN rows sampled (seed 42) from the unified TRAIN split, shared by
both explainers. The AE is trained on BENIGN only and the IF's notion of normal is the bulk of
the data, so "a typical benign flow" is the right reference for both: an attribution answers
"which feature values push this row away from normal".

RISK SCORE. Matches src/notifications/alert_dispatcher.assemble_alert_record
(`round(fused * 100)`), clamped to 0-100 here because the fused score is intentionally
unclipped and the dispatcher does not clamp. The two agree for every fused_score <= 1.0.

Feature values reported are model-input values (MinMax-scaled), not raw packet counts.

Run the report: python -m src.modeling.shap_explain
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from src.modeling import fusion_layer

ROOT = Path(__file__).resolve().parents[2]

NON_FEATURE_COLS = [
    "Label", "attack_cat", "technique_id", "technique_name", "tactic", "mapping_confidence",
    "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique",
    "orig_row_index",
]

RF_PATH = ROOT / "models/random_forest_unified.pkl"
IF_PATH = ROOT / "models/isolation_forest_unified.pkl"
AE_PATH = ROOT / "models/autoencoder_unified.pt"
AE_CONFIG_PATH = ROOT / "models/autoencoder_unified_config.json"
TRAIN_PATH = ROOT / "data/features/unified/cicids2017_unified_train.parquet"
TEST_PATH = ROOT / "data/features/unified/cicids2017_unified_test.parquet"
REPORT_PATH = ROOT / "reports/shap_explainability_report.txt"
IMPORTANCE_PATH = ROOT / "reports/shap_global_importance.csv"

BACKGROUND_ROWS = 100
KERNEL_NSAMPLES = 500
EXPECTED_FEATURES = 94
SEED = 42

# Guard against the module silently serving a strategy it was not written for -- the failure
# it replaced was exactly that, and it produced attributions that looked fine.
WRITTEN_FOR_STRATEGY = "confidence_weighted"
if fusion_layer.CHOSEN_STRATEGY != WRITTEN_FOR_STRATEGY:
    raise RuntimeError(
        f"shap_explain is written for fusion strategy '{WRITTEN_FOR_STRATEGY}' but the deployed "
        f"strategy is '{fusion_layer.CHOSEN_STRATEGY}'. Attributions would explain a decision "
        f"rule that is not in use. Migrate this module before running it."
    )

ModelUsed = Literal["random_forest", "if_ae_fallback"]
assert set(fusion_layer.MODEL_USED_VALUES) == {"random_forest", "if_ae_fallback"}


class FeatureAttribution(TypedDict):
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class Explanation(TypedDict):
    risk_score: int
    model_used: ModelUsed
    fused_score: float
    top_features: list[FeatureAttribution]


def numeric_feature_columns(df: pd.DataFrame, excluded: list[str] = None) -> list[str]:
    excluded = NON_FEATURE_COLS if excluded is None else excluded
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def risk_score_from_fused(fused_score: float) -> int:
    """Dispatcher's derivation (`round(fused * 100)`), clamped to 0-100 for display."""
    return int(min(100, max(0, round(float(fused_score) * 100))))


class _Autoencoder(nn.Module):
    """Same architecture as train_unified_ensemble.py and autoencoder_unified_config.json."""

    def __init__(self, d, hidden, bottleneck):
        super().__init__()
        enc_dims = [d] + list(hidden) + [bottleneck]
        dec_dims = [bottleneck] + list(hidden)[::-1] + [d]

        def stack(dims):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.encoder, self.decoder = stack(enc_dims), stack(dec_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _top_k(names, values, shap_vals, k) -> list[FeatureAttribution]:
    order = np.argsort(-np.abs(shap_vals))[:k]
    return [
        FeatureAttribution(
            feature=names[i],
            value=float(values[i]),
            shap_value=float(shap_vals[i]),
            direction="increases_risk" if shap_vals[i] > 0 else "decreases_risk",
        )
        for i in order
    ]


class FusionExplainer:
    """Loads the three deployed unified models once; explains rows by the branch that decided them."""

    def __init__(self, background: pd.DataFrame | None = None):
        self.rf = joblib.load(RF_PATH)
        self.benign_idx = list(self.rf.classes_).index("BENIGN")
        self.iso = joblib.load(IF_PATH)

        cfg = json.loads(AE_CONFIG_PATH.read_text(encoding="utf-8"))
        self.ae = _Autoencoder(cfg["input_dim"], cfg["hidden_dims"], cfg["bottleneck_dim"])
        self.ae.load_state_dict(torch.load(AE_PATH, map_location="cpu"))
        self.ae.eval()

        if background is None:
            background = self._load_background()
        self.columns = numeric_feature_columns(background)
        assert len(self.columns) == EXPECTED_FEATURES, \
            f"expected {EXPECTED_FEATURES} features, got {len(self.columns)}"
        assert len(self.columns) == cfg["input_dim"], \
            f"AE expects {cfg['input_dim']} inputs, population has {len(self.columns)}"
        self.background = background[self.columns].to_numpy(dtype=np.float64)

        self.rf_tree = shap.TreeExplainer(self.rf)
        # One KernelExplainer over the composite the cascade actually thresholds.
        self.fallback_kernel = shap.KernelExplainer(self._fallback_fn, self.background)

    @staticmethod
    def _load_background() -> pd.DataFrame:
        train = pd.read_parquet(TRAIN_PATH)
        benign = train[train["Label"] == "BENIGN"]
        return benign.sample(BACKGROUND_ROWS, random_state=SEED)

    # ── member scores, computed exactly as the deployed code does ────
    def _ae_mse(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, dtype=np.float32))
            return torch.mean((self.ae(t) - t) ** 2, dim=1).numpy()

    def _if_raw(self, x: np.ndarray) -> np.ndarray:
        return -self.iso.score_samples(np.asarray(x, dtype=np.float64))

    def _fallback_fn(self, x: np.ndarray) -> np.ndarray:
        """The cascade's fallback score: the mean of both normalized unsupervised scores.

        This is the function KernelExplainer attributes, so the SHAP values explain the
        quantity that set the row's fused score rather than one contributor to it.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if_norm = fusion_layer.normalize_if_score(self._if_raw(x))
        ae_norm = fusion_layer.normalize_ae_score(self._ae_mse(x))
        return (if_norm + ae_norm) / 2.0

    def _rf_attack_shap(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """(n, features) SHAP of rf_attack_probability = 1 - P(BENIGN), and its base value."""
        raw = self.rf_tree.shap_values(x, check_additivity=False)
        benign = raw[self.benign_idx] if isinstance(raw, list) else raw[:, :, self.benign_idx]
        base = np.atleast_1d(self.rf_tree.expected_value)[self.benign_idx]
        return -benign, float(1.0 - base)

    # ── public API ───────────────────────────────────────────────────
    def explain(self, rows: pd.DataFrame, top_k: int = 5) -> list[Explanation]:
        """Explain a batch of rows from the unified population.

        One frame, one feature space -- the predecessor took two frames and an
        `orig_row_index` join because its members lived in different populations.
        """
        x = rows[self.columns].to_numpy(dtype=np.float64)
        rf_prob = 1.0 - self.rf.predict_proba(x)[:, self.benign_idx]
        if_raw, ae_err = self._if_raw(x), self._ae_mse(x)
        fused, _, _, model_used = fusion_layer.fuse_with_metadata(rf_prob, if_raw, ae_err)

        records: list[Explanation | None] = [None] * len(fused)
        rf_rows = np.flatnonzero(model_used == "random_forest")
        fb_rows = np.flatnonzero(model_used == "if_ae_fallback")

        if len(rf_rows):
            sv, _ = self._rf_attack_shap(x[rf_rows])
            for j, i in enumerate(rf_rows):
                records[i] = self._record(fused[i], "random_forest", self.columns, x[i], sv[j], top_k)
        for i in fb_rows:
            sv = np.asarray(
                self.fallback_kernel.shap_values(x[i:i + 1], nsamples=KERNEL_NSAMPLES, silent=True)
            ).reshape(-1)
            records[i] = self._record(fused[i], "if_ae_fallback", self.columns, x[i], sv, top_k)
        assert all(r is not None for r in records), "a row was decided by no known branch"
        return records  # type: ignore[return-value]

    @staticmethod
    def _record(fused, model_used, names, values, shap_vals, k) -> Explanation:
        return Explanation(
            risk_score=risk_score_from_fused(fused),
            model_used=model_used,
            fused_score=float(fused),
            top_features=_top_k(names, values, shap_vals, k),
        )

    def additivity_gap(self, member: ModelUsed, rows: pd.DataFrame) -> float:
        """Max |base + sum(shap) - the branch's own score| over the rows. A faithfulness check.

        If this is not tiny, the attribution does not explain the deployed score and the
        numbers in the report are decoration.
        """
        x = rows[self.columns].to_numpy(dtype=np.float64)
        if member == "random_forest":
            sv, base = self._rf_attack_shap(x)
            target = 1.0 - self.rf.predict_proba(x)[:, self.benign_idx]
            return float(np.max(np.abs(base + sv.sum(1) - target)))
        sv = np.asarray(
            self.fallback_kernel.shap_values(x, nsamples=KERNEL_NSAMPLES, silent=True)
        ).reshape(len(x), -1)
        base = float(np.atleast_1d(self.fallback_kernel.expected_value)[0])
        return float(np.max(np.abs(base + sv.sum(1) - self._fallback_fn(x))))


# ----------------------------------------------------------------------------
# Report generation: python -m src.modeling.shap_explain
# ----------------------------------------------------------------------------
GLOBAL_RF_ROWS = 300
GLOBAL_FALLBACK_ROWS = 60   # KernelExplainer over the composite is the slow path
LATENCY_ROWS = 60


def _balanced_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    attack = frame[frame["Label"] != "BENIGN"]
    benign = frame[frame["Label"] == "BENIGN"]
    return pd.concat(
        [attack.sample(n // 2, random_state=seed), benign.sample(n - n // 2, random_state=seed)]
    )


def _split_by_branch(explainer: FusionExplainer, frame: pd.DataFrame):
    """Partition rows by the branch that decides them, and report the measured mix.

    The fallback branch fires on roughly 0.2% of in-dataset rows, so a random sample would
    almost never contain one and that path would go unmeasured. It is therefore sampled
    deliberately. Stating the mix matters more than the convenience: RF is decisive on
    virtually all in-dataset traffic, while on the synthetic overlay -- out-of-distribution
    input -- RF abstained on every flow and the fallback carried all of it.
    """
    x = frame[explainer.columns].to_numpy(dtype=np.float64)
    rf_prob = 1.0 - explainer.rf.predict_proba(x)[:, explainer.benign_idx]
    decisive = np.abs(rf_prob - 0.5) * 2 >= fusion_layer.CONFIDENCE_THRESHOLD
    return frame[decisive], frame[~decisive], float(decisive.mean())


def main() -> None:
    import time

    test = pd.read_parquet(TEST_PATH)
    explainer = FusionExplainer()
    cols = explainer.columns

    # Find real rows of each branch. Scoring a 60k slice is enough to collect a fallback
    # sample without scoring all 504k.
    scan = test.sample(60_000, random_state=SEED)
    rf_pool, fb_pool, rf_share = _split_by_branch(explainer, scan)
    fb_available = len(fb_pool)
    rf_sample = _balanced_sample(rf_pool, min(GLOBAL_RF_ROWS, len(rf_pool)), SEED)
    fb_sample = fb_pool.sample(min(GLOBAL_FALLBACK_ROWS, fb_available), random_state=SEED + 1)

    # ── global importance per branch ─────────────────────────────────
    rf_sv, _ = explainer._rf_attack_shap(rf_sample[cols].to_numpy(dtype=np.float64))
    fb_sv = np.asarray(
        explainer.fallback_kernel.shap_values(
            fb_sample[cols].to_numpy(dtype=np.float64), nsamples=KERNEL_NSAMPLES, silent=True
        )
    ).reshape(len(fb_sample), -1)
    importance = pd.concat([
        pd.DataFrame({"member": "random_forest", "feature": cols,
                      "mean_abs_shap": np.abs(rf_sv).mean(0)}),
        pd.DataFrame({"member": "if_ae_fallback", "feature": cols,
                      "mean_abs_shap": np.abs(fb_sv).mean(0)}),
    ]).sort_values(["member", "mean_abs_shap"], ascending=[True, False])
    importance.to_csv(IMPORTANCE_PATH, index=False)

    # ── faithfulness ─────────────────────────────────────────────────
    gaps = {
        "random_forest": explainer.additivity_gap("random_forest", rf_sample.iloc[:20]),
        "if_ae_fallback": explainer.additivity_gap("if_ae_fallback", fb_sample.iloc[:20]),
    }

    # ── latency and branch mix ───────────────────────────────────────
    # Half from each branch, so both latencies are actually measured.
    half = LATENCY_ROWS // 2
    latency_sample = pd.concat([
        rf_pool.sample(min(half, len(rf_pool)), random_state=SEED + 2),
        fb_pool.sample(min(half, fb_available), random_state=SEED + 2),
    ])
    timings: dict[str, list[float]] = {"random_forest": [], "if_ae_fallback": []}
    chosen: list[Explanation] = []
    for i in range(len(latency_sample)):
        row = latency_sample.iloc[i:i + 1]
        start = time.perf_counter()
        rec = explainer.explain(row, top_k=5)[0]
        timings[rec["model_used"]].append(time.perf_counter() - start)
        chosen.append(rec)
    decided = pd.Series([r["model_used"] for r in chosen]).value_counts().to_dict()

    bar = "=" * 74
    L = [bar, "SHAP EXPLAINABILITY OVER THE DEPLOYED FUSION LAYER", bar,
         "Command: python -m src.modeling.shap_explain", ""]
    L += [bar, "1. WHAT IS ATTRIBUTED", bar,
          f"Deployed strategy: {fusion_layer.CHOSEN_STRATEGY} "
          f"(confidence {fusion_layer.CONFIDENCE_THRESHOLD}, "
          f"threshold {fusion_layer.FUSED_DECISION_THRESHOLD}).",
          "Each row is decided by one branch, named by fusion_layer.fuse_with_metadata, and only",
          "that branch is attributed -- in the units of the fused score, so base + sum(shap)",
          "reconstructs the row's fused_score.",
          "  random_forest   TreeExplainer on 1 - P(BENIGN), which IS the fused score on these rows.",
          "  if_ae_fallback  KernelExplainer on the COMPOSITE (if_norm + ae_norm) / 2, the quantity",
          "                  the cascade thresholds. Both members share one 94-column feature space",
          f"                  on the unified population, which is what makes this attributable;",
          "                  under the retired two-population design it was not.",
          f"Feature space: {len(cols)} columns, identical for all three members.",
          f"Background: {BACKGROUND_ROWS} BENIGN rows from the unified TRAIN split (seed {SEED}).", ""]
    L += [bar, "2. FAITHFULNESS (max |base + sum(shap) - branch score|)", bar]
    for k, v in gaps.items():
        L.append(f"  {k:16s} {v:.3e}")
    L += ["If either number were not tiny, the attributions would not explain the deployed score.", ""]
    L += [bar, "3. MEASURED LATENCY PER ALERT", bar]
    for k, v in timings.items():
        if v:
            L.append(f"  {k:16s} n={len(v):3d}  mean {np.mean(v):.3f}s  median {np.median(v):.3f}s")
        else:
            L.append(f"  {k:16s} no rows in the latency sample took this branch")
    # Which path is slower is MEASURED, not asserted. A hand-written claim here was wrong on
    # the first run: KernelExplainer over the composite looks like it should dominate, but the
    # TreeExplainer pass over 200 trees x 15 classes costs more.
    means = {k: float(np.mean(v)) for k, v in timings.items() if v}
    if len(means) == 2:
        slow, fast = max(means, key=means.get), min(means, key=means.get)
        L.append(f"The slower path here is {slow} ({means[slow]:.3f}s vs {means[fast]:.3f}s).")
        if slow == "random_forest":
            L += ["That is the opposite of what the shapes suggest: KernelExplainer evaluates the",
                  f"composite over {KERNEL_NSAMPLES} coalitions against {BACKGROUND_ROWS} background",
                  "rows, running an Isolation Forest pass and an AE forward pass each time, yet the",
                  "TreeExplainer pass over 200 trees x 15 output classes still costs more."]
        else:
            L += [f"KernelExplainer evaluates the composite over {KERNEL_NSAMPLES} coalitions against",
                  f"{BACKGROUND_ROWS} background rows, running an Isolation Forest pass and an AE",
                  "forward pass each time, which is why it dominates."]
    L += ["Both are usable per alert and impractical for bulk scoring -- unchanged conclusion.", ""]
    L += [bar, "4. BRANCH MIX", bar,
          f"Measured on a random 60,000-row slice of the unified test split:",
          f"  random_forest decisive : {rf_share * 100:.3f}%",
          f"  if_ae_fallback         : {(1 - rf_share) * 100:.3f}%  ({fb_available:,} rows)",
          "The fallback is rare in-dataset, so it is sampled DELIBERATELY above rather than at",
          "random -- a random sample of 60 rows would contain none and the path would go",
          "unmeasured. The contrast is the interesting part: on the synthetic overlay, which is",
          "out-of-distribution input, RF abstained on every flow and the fallback decided 100% of",
          "the alerts (reports/mttd_report.txt). The fallback exists for that case.",
          f"Latency sample, after deliberate stratification: {decided}", ""]
    L += [bar, "5. TOP GLOBAL FEATURES PER BRANCH (mean |SHAP|)", bar]
    for member in ("random_forest", "if_ae_fallback"):
        sub = importance[importance["member"] == member].head(10)
        n = len(rf_sample) if member == "random_forest" else len(fb_sample)
        L.append(f"{member} (n={n} rows)")
        L.append(sub[["feature", "mean_abs_shap"]].to_string(index=False,
                                                             float_format=lambda v: f"{v:.6f}"))
        L.append("")
    L += [bar, "6. CAVEATS", bar,
          "Feature values in a record are MinMax-scaled model inputs, not raw packet counts;",
          "translate before showing an analyst.",
          "Global importance is a sample statistic over the rows listed above, not over the full",
          "test split, and the fallback sample is small because that path is slow.",
          "A high mean |SHAP| means a feature moves this model's score, not that it is causal.",
          ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[saved] {REPORT_PATH}")
    print(f"[saved] {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
