"""Build ONE population and ONE feature set for all three ensemble members.

Why this exists. The thesis advisor rejected the fusion on the grounds that the supervised
and unsupervised members are not on the same dataset, and she is right about the mechanism.
Random Forest reads the `MachineLearningCSV`-derived matrix while IF-v2 and AE-v2 read the
`GeneratedLabelledFlows`-derived behavioral matrix. Those are two independent CICFlowMeter
extractions of the same captures, so `validate_behavioral_join.py` has to row-hash-join them
back together, and that join matches only 76.4% of BENIGN rows. The ensemble can therefore
only be evaluated on 80.36% of the test split, and that 405,142-row population is an artifact
of joining two CSVs rather than anything a deployed system would ever see.

The fix is to stop joining. `cicids2017_behavioral_labeled.parquet` already carries every
column all three members need (the raw CICFlowMeter features, the 12 behavioral columns, the
MITRE reference columns) for 2,520,961 rows, which is slightly MORE than the main pipeline's
own cleaned population. Splitting directly from it gives one population at 100% coverage, one
feature set, and no join.

What this costs, stated up front: every number measured on the old splits is superseded.
RF in particular now trains on the behavioral population rather than the `MachineLearningCSV`
one, so its solo baseline will move. That is the honest trade. Comparing an ensemble that saw
80.36% of the data against an RF that saw 100% was never a fair comparison.

Run from the repo root with the Anaconda interpreter:
    python -m src.features.build_unified_population
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import feature_extractor as FE
from src.ingestion import preprocess_pipeline as PP

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/behavioral/cicids2017_behavioral_labeled.parquet"
OUT_DIR = ROOT / "data/features/unified"
LOAO_DIR = ROOT / "data/processed/unified_loao"
REPORT_PATH = ROOT / "reports/build_unified_population_report.txt"

RANDOM_STATE = 42
SENTINEL_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
REFERENCE_COLS = ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"]
# Identifiers are dropped from model input on purpose. Source IP in particular would let a
# model memorise which hosts attacked in this capture, which is leakage dressed as a feature.
# The behavioral columns already carry what the identifiers were needed for.
IDENTIFIER_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "day_group"]

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 70)
    log(title)
    log("=" * 70)


def split_population(df: pd.DataFrame):
    """Stratified 60/20/20 on Label, the same proportions and seed the main pipeline uses."""
    train, temp = train_test_split(
        df, test_size=0.4, random_state=RANDOM_STATE, stratify=df["Label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=RANDOM_STATE, stratify=temp["Label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_split_features(train_df, val_df, test_df, network_cols):
    """Impute, encode, scale and resample exactly as feature_extractor does for the main path.

    Every fitted quantity comes from train only: the sentinel medians, the CTI integer
    encodings, the scaler, and the SMOTE resampling. Val and test are transformed, never fit.
    """
    medians = FE.fit_sentinel_imputation(train_df, SENTINEL_COLS)
    frames = {
        "train": FE.apply_sentinel_imputation(train_df, SENTINEL_COLS, medians),
        "val": FE.apply_sentinel_imputation(val_df, SENTINEL_COLS, medians),
        "test": FE.apply_sentinel_imputation(test_df, SENTINEL_COLS, medians),
    }
    flag_cols = [f"{c}_missing" for c in SENTINEL_COLS]

    technique_map = FE.fit_integer_encoding(frames["train"]["technique_id"])
    tactic_map = FE.fit_integer_encoding(frames["train"]["tactic"])
    for name, frame in frames.items():
        frame["technique_id_encoded"] = FE.apply_integer_encoding(
            frame["technique_id"], technique_map, "cicids2017_unified", "technique_id", name
        )
        frame["tactic_encoded"] = FE.apply_integer_encoding(
            frame["tactic"], tactic_map, "cicids2017_unified", "tactic", name
        )
        frame["mapping_confidence_encoded"] = FE.apply_mapping_confidence_encoding(
            frame["mapping_confidence"]
        )
        frame["is_known_technique"] = FE.build_is_known_technique(frame["technique_id"])

    scaler = FE.fit_scaler(frames["train"], network_cols)
    for name, frame in frames.items():
        frames[name] = FE.apply_scaler(frame, network_cols, scaler)

    model_cols = network_cols + flag_cols
    log(f"  model input features: {len(model_cols)} "
        f"({len(network_cols)} network + {len(flag_cols)} sentinel flags)")

    X = frames["train"][model_cols].to_numpy(dtype=np.float64)
    # apply_smote counts classes with pandas value_counts and returns a third applied flag
    y = frames["train"]["Label"].reset_index(drop=True)
    X_res, y_res, smote_applied = FE.apply_smote(X, y, "cicids2017_unified")
    log(f"  SMOTE applied: {smote_applied}")
    log(f"  train rows: {len(X):,} -> {len(X_res):,} after capped SMOTE")

    resampled = pd.DataFrame(X_res, columns=model_cols)
    resampled["Label"] = y_res.to_numpy() if hasattr(y_res, "to_numpy") else y_res
    # SMOTE invents rows, so the MITRE reference columns are re-derived from Label rather
    # than carried through: a synthetic row has no annotation of its own. feature_extractor
    # already solves this, including the warning for labels whose reference values are not
    # unique, so reuse it instead of hand-rolling the mapping.
    non_label_refs = [c for c in REFERENCE_COLS if c != "Label"]
    ref_map, non_deterministic = FE.build_label_reference_map(
        frames["train"], "Label", non_label_refs
    )
    ref_df = FE.attach_reference_columns(
        resampled["Label"], ref_map, non_label_refs, non_deterministic, "cicids2017_unified"
    )
    for col in non_label_refs:
        resampled[col] = ref_df[col].to_numpy()
    resampled["technique_id_encoded"] = FE.apply_integer_encoding(
        resampled["technique_id"], technique_map, "cicids2017_unified", "technique_id", "train_smote"
    )
    resampled["tactic_encoded"] = FE.apply_integer_encoding(
        resampled["tactic"], tactic_map, "cicids2017_unified", "tactic", "train_smote"
    )
    resampled["mapping_confidence_encoded"] = FE.apply_mapping_confidence_encoding(
        resampled["mapping_confidence"]
    )
    resampled["is_known_technique"] = FE.build_is_known_technique(resampled["technique_id"])

    keep = model_cols + REFERENCE_COLS + [
        "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique"
    ]
    out = {"train": resampled[keep]}
    for name in ("val", "test"):
        out[name] = frames[name][keep].reset_index(drop=True)
    return out, model_cols


def main() -> None:
    section("STEP 1: load the single source population")
    df = pd.read_parquet(SOURCE)
    log(f"  {SOURCE.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    drop = [c for c in IDENTIFIER_COLS if c in df.columns]
    network_cols = [
        c for c in df.columns
        if c not in drop and c not in REFERENCE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    log(f"  identifiers dropped from model input: {drop}")
    log(f"  network features: {len(network_cols)}")
    behavioral = [c for c in network_cols if c.endswith(("_1m", "_3m", "_5m"))]
    log(f"  of which behavioral: {len(behavioral)}")
    assert len(behavioral) == 12, f"expected 12 behavioral columns, got {len(behavioral)}"

    section("STEP 2: stratified 60/20/20 split, no join anywhere")
    train_df, val_df, test_df = split_population(df)
    total = len(train_df) + len(val_df) + len(test_df)
    log(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}  total {total:,}")
    assert total == len(df), "split lost rows"
    log(f"  coverage of the source population: {total / len(df) * 100:.2f}%")
    log("  the old path joined two CICFlowMeter extractions and kept 80.36%; this keeps all of it")

    section("STEP 3: impute, encode, scale, resample (all fitted on train only)")
    frames, model_cols = build_split_features(train_df, val_df, test_df, network_cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = OUT_DIR / f"cicids2017_unified_{name}.parquet"
        frame.to_parquet(path, index=False)
        log(f"  wrote {path.name}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")

    section("STEP 4: LOAO zero-day splits from the same population")
    LOAO_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["Label"].value_counts()
    eligible = [c for c in counts.index if c != "BENIGN" and counts[c] >= PP.LOAO_MIN_SAMPLES]
    log(f"  categories at or above n={PP.LOAO_MIN_SAMPLES}: {len(eligible)}")
    for category in eligible:
        fold_train, fold_test = PP.build_loao_split(df, category)
        slug = PP.slugify_category(category) if hasattr(PP, "slugify_category") else \
            category.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        fold_train.to_parquet(LOAO_DIR / f"loao_{slug}_train.parquet", index=False)
        fold_test.to_parquet(LOAO_DIR / f"loao_{slug}_zeroday_test.parquet", index=False)
        log(f"    {category:28s} train {len(fold_train):>9,}  zero-day test {len(fold_test):>9,}")

    section("STEP 5: what this supersedes")
    log("  Every metric measured on data/features/cicids2017_features_* or")
    log("  data/features/behavioral/cicids2017_behavioral_features_* is now on a different")
    log("  population and is not comparable to anything produced from these files.")
    log("  Rerun, in order: the three member trainings, the LOAO threshold selection,")
    log("  eval_test_set_ensemble, shap_explain, and siem_baseline.")
    log("  M1c is NOT unblocked by this. CIC-UNSW-NB15's Data.csv carries no Source IP or")
    log("  Timestamp, so the 12 behavioral columns cannot be computed for it and the")
    log("  unsupervised members still cannot take part in cross-dataset evaluation.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n[saved] {REPORT_PATH}")


if __name__ == "__main__":
    main()
