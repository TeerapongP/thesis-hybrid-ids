from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import imblearn
import sklearn
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.features import build_unified_population as BUP
from src.modeling import fusion_layer as FL
from src.modeling import train_unified_ensemble as TUE

ROOT = Path(__file__).resolve().parents[2]
SOURCE = BUP.SOURCE
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
AE_CONFIG_PATH = MODELS_DIR / "autoencoder_behavioral_v2_config.json"
FUSION_CONFIG_PATH = MODELS_DIR / "fusion_config.json"

RANDOM_STATE = TUE.RANDOM_STATE
OUTER_SPLITS = 5
INNER_SPLITS = 4
GROUP_BLOCK = "5min"
MIN_ROWS_PER_CLASS_SMOKE = 20
FOLD_ALGORITHM_VERSION = 1
MODEL_NAMES = ("rf", "if", "ae", "fused")
METRIC_NAMES = ("precision", "recall", "f1", "auc_roc", "fpr")
SCHEMES = {"random": StratifiedKFold, "grouped": StratifiedGroupKFold}

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 74)
    log(title)
    log("=" * 74)


def checkpoint_root(smoke: bool) -> Path:
    return REPORTS / f"split_robustness_checkpoints{'_smoke' if smoke else ''}"


def output_paths(smoke: bool, run_id: str) -> dict[str, Path]:
    suffix = "_smoke" if smoke else ""
    return {
        "folds": REPORTS / f"split_robustness_folds{suffix}_{run_id}.csv",
        "per_class": REPORTS / f"split_robustness_per_class{suffix}_{run_id}.csv",
        "summary": REPORTS / f"split_robustness_summary{suffix}_{run_id}.json",
        "report": REPORTS / f"split_robustness_report{suffix}_{run_id}.txt",
    }


def run_id_for(signatures) -> str:
    return hashlib.sha256(",".join(sorted(signatures)).encode()).hexdigest()[:10]


def make_splitter(scheme: str, n_splits: int):
    return SCHEMES[scheme](n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def iter_splits(scheme: str, n_splits: int, labels, groups):
    pass_groups = groups if scheme == "grouped" else None
    return make_splitter(scheme, n_splits).split(np.zeros(len(labels)), labels, pass_groups)


def assign_outer_folds(labels, groups, scheme: str, n_splits: int = OUTER_SPLITS) -> np.ndarray:
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    fold_ids = np.full(len(labels), -1, dtype=int)
    for k, (_, test_idx) in enumerate(iter_splits(scheme, n_splits, labels, groups)):
        fold_ids[test_idx] = k
    return fold_ids


def split_fold(labels, groups, outer_fold_ids, k: int, scheme: str):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    outer_fold_ids = np.asarray(outer_fold_ids)
    test_idx = np.flatnonzero(outer_fold_ids == k)
    remaining = np.flatnonzero(outer_fold_ids != k)
    train_local, val_local = next(
        iter_splits(scheme, INNER_SPLITS, labels[remaining], groups[remaining])
    )
    return remaining[train_local], remaining[val_local], test_idx


def build_group_ids(df: pd.DataFrame) -> np.ndarray:
    block = df["Timestamp"].dt.floor(GROUP_BLOCK).astype(str)
    key = df["day_group"].astype(str) + "|" + df["Source IP"].astype(str) + "|" + block
    return pd.factorize(key)[0]


def pct_rows_sharing_groups(query_idx, reference_idx, groups) -> float:
    if len(query_idx) == 0:
        return 0.0
    shared = np.isin(groups[query_idx], np.unique(groups[reference_idx]))
    return float(100.0 * shared.mean())


def class_counts(labels, idx) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels)[idx], return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def held_out_classes(train_counts: dict, test_counts: dict) -> dict[str, int]:
    return {label: n for label, n in test_counts.items() if train_counts.get(label, 0) == 0}


def check_train_class_counts(train_counts: dict) -> None:
    singles = sorted(label for label, n in train_counts.items() if n == 1)
    if singles:
        raise ValueError(
            f"classes with exactly 1 train row make SMOTE in FE.apply_smote fail: {singles}"
        )


def derive_network_cols(df: pd.DataFrame) -> list[str]:
    drop = [c for c in BUP.IDENTIFIER_COLS if c in df.columns]
    cols = [
        c for c in df.columns
        if c not in drop and c not in BUP.REFERENCE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    behavioral = [c for c in cols if c.endswith(("_1m", "_3m", "_5m"))]
    assert len(behavioral) == 12, f"expected 12 behavioral columns, got {len(behavioral)}"
    return cols


def stratified_subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows >= len(df):
        return df
    frac = max_rows / len(df)
    parts = []
    for _, part in df.groupby("Label", sort=True):
        take = min(len(part), max(int(math.ceil(len(part) * frac)), MIN_ROWS_PER_CLASS_SMOKE))
        parts.append(part.sample(n=take, random_state=RANDOM_STATE))
    return pd.concat(parts).sort_index()


def load_population(max_rows: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(SOURCE)
    df["Label"] = df["Label"].astype(str)
    if max_rows is not None:
        df = stratified_subsample(df, max_rows)
    source_row = df.index.to_numpy(dtype=np.int64)
    return df.reset_index(drop=True), source_row


def array_hash(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()


def data_fingerprint(df: pd.DataFrame, network_cols, source_row, max_rows) -> dict:
    stat = SOURCE.stat()
    label_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df["Label"], index=False).to_numpy().tobytes()
    ).hexdigest()
    return {
        "n_rows": int(len(df)),
        "label_hash": label_hash,
        "source_row_hash": array_hash(source_row),
        "network_cols": list(network_cols),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "max_rows": max_rows,
        "min_rows_per_class_smoke": MIN_ROWS_PER_CLASS_SMOKE,
        "smote_flat_cap": BUP.FE.SMOTE_FLAT_CAP,
        "smote_target_multiplier": BUP.FE.SMOTE_TARGET_MULTIPLIER,
        "smote_random_state": BUP.FE.SMOTE_RANDOM_STATE,
        "sklearn": sklearn.__version__,
        "imblearn": imblearn.__version__,
    }


def run_signature(scheme: str, fold: int, test_idx, fingerprint: dict, ae_cfg: dict, fusion_cfg: dict) -> str:
    payload = {
        "fold_algorithm_version": FOLD_ALGORITHM_VERSION,
        "scheme": scheme,
        "fold": fold,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "group_block": GROUP_BLOCK,
        "test_idx_hash": array_hash(test_idx),
        "data": fingerprint,
        "confidence_threshold": fusion_cfg["confidence_threshold"],
        "fused_decision_threshold": fusion_cfg["fused_decision_threshold"],
        "ae": {k: ae_cfg.get(k) for k in ("hidden_dims", "bottleneck_dim", "activation",
                                          "learning_rate", "epochs", "batch_size",
                                          "weight_decay", "early_stopping_patience")},
        "rf": {"n_estimators": TUE.RF_N_ESTIMATORS, "class_weight": TUE.RF_CLASS_WEIGHT},
        "if_contamination": TUE.IF_CONTAMINATION,
        "ae_percentile": TUE.AE_PERCENTILE,
        "ae_benign_val_frac": TUE.AE_BENIGN_VAL_FRAC,
        "random_state": RANDOM_STATE,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def checkpoint_paths(root: Path, signature: str, scheme: str, fold: int) -> tuple[Path, Path]:
    stem = f"{scheme}_fold{fold}"
    directory = root / signature
    return directory / f"{stem}.json", directory / f"{stem}_test_outputs.parquet"


def load_checkpoint(root: Path, scheme: str, fold: int, signature: str, test_idx):
    meta_path, rows_path = checkpoint_paths(root, signature, scheme, fold)
    if not (meta_path.exists() and rows_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = pd.read_parquet(rows_path)
    except Exception:
        return None
    if meta.get("run_signature") != signature or rows.empty:
        return None
    if not np.array_equal(rows["row_id"].to_numpy(), np.asarray(test_idx)):
        return None
    return meta, rows


def save_checkpoint(root: Path, meta: dict, rows: pd.DataFrame) -> None:
    meta_path, rows_path = checkpoint_paths(root, meta["run_signature"], meta["scheme"], meta["fold"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_path, index=False)
    meta_path.write_text(json.dumps(clean(meta), indent=2), encoding="utf-8")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    return value


def model_metrics(y_true, pred, score) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    benign = y_true == 0
    fpr = float(pred[benign].mean()) if benign.any() else float("nan")
    if len(np.unique(y_true)) < 2:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "auc_roc": float("nan"), "fpr": fpr}
    base = TUE.binary_metrics(y_true, pred, score)
    return {**{k: float(v) for k, v in base.items()}, "fpr": fpr}


def per_class_recall(labels, y_true, pred, seen) -> dict:
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    seen = np.asarray(seen, dtype=bool)
    out = {}
    for name in sorted(set(labels[np.asarray(y_true) == 1])):
        mask = labels == name
        seen_mask = mask & seen
        out[name] = {
            "n": int(mask.sum()),
            "recall": float(pred[mask].mean()),
            "n_seen": int(seen_mask.sum()),
            "recall_seen": float(pred[seen_mask].mean()) if seen_mask.any() else float("nan"),
        }
    return out


def seen_mask(rows: pd.DataFrame, exclude_ids=None) -> np.ndarray:
    if exclude_ids is None:
        return rows["class_in_train"].to_numpy(dtype=bool)
    return ~rows["row_id"].isin(exclude_ids).to_numpy()


def excluded_rows(results: dict) -> tuple[np.ndarray, dict]:
    parts = [rows.loc[~rows["class_in_train"], ["row_id", "Label"]]
             for entries in results.values() for _, rows in entries]
    if not parts:
        return np.array([], dtype=np.int64), {}
    union = pd.concat(parts).drop_duplicates("row_id")
    return union["row_id"].to_numpy(), {str(k): int(v) for k, v in union["Label"].value_counts().items()}


def evaluate_rows(rows: pd.DataFrame, exclude_ids=None, seen_only: bool = False) -> tuple[dict, dict]:
    seen = seen_mask(rows, exclude_ids)
    if seen_only:
        rows = rows[seen]
        seen = np.ones(len(rows), dtype=bool)
    y_true = rows["y_true"].to_numpy()
    metrics, per_class = {}, {}
    for model in MODEL_NAMES:
        pred = rows[f"{model}_pred"].to_numpy()
        metrics[model] = model_metrics(y_true, pred, rows[f"{model}_score"].to_numpy())
        per_class[model] = per_class_recall(
            rows["Label"].to_numpy(), y_true, pred, seen
        )
    return metrics, per_class


def fit_and_score_members(frames: dict, model_cols: list[str], ae_cfg: dict, fusion_cfg: dict) -> pd.DataFrame:
    train, test = frames["train"], frames["test"]
    X_train = train[model_cols].to_numpy(dtype=np.float64)
    X_test = test[model_cols].to_numpy(dtype=np.float64)
    y_train_label = train["Label"].to_numpy()
    test_labels = test["Label"].to_numpy()
    y_true = (test_labels != "BENIGN").astype(int)

    rf = RandomForestClassifier(
        n_estimators=TUE.RF_N_ESTIMATORS, class_weight=TUE.RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train_label)
    proba = rf.predict_proba(X_test)
    benign_idx = list(rf.classes_).index("BENIGN")
    rf_score = 1.0 - proba[:, benign_idx]
    rf_pred = (rf.classes_[proba.argmax(1)] != "BENIGN").astype(int)
    del rf

    iso = IsolationForest(contamination=TUE.IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)
    if_raw_train = -iso.score_samples(X_train)
    if_raw_test = -iso.score_samples(X_test)
    if_pred = (iso.predict(X_test) == -1).astype(int)

    benign_mask = y_train_label == "BENIGN"
    X_benign = X_train[benign_mask].astype(np.float32)
    fit_X, val_X = train_test_split(
        X_benign, test_size=TUE.AE_BENIGN_VAL_FRAC, random_state=RANDOM_STATE
    )
    ae, _ = TUE.train_autoencoder(fit_X, val_X, len(model_cols), ae_cfg)
    ae_internal = TUE.reconstruction_error(ae, val_X)
    ae_threshold = float(np.percentile(ae_internal, TUE.AE_PERCENTILE))
    ae_raw_train = TUE.reconstruction_error(ae, X_train.astype(np.float32))
    ae_raw_test = TUE.reconstruction_error(ae, X_test.astype(np.float32))
    ae_pred = (ae_raw_test > ae_threshold).astype(int)

    if_norm = FL.normalize_if_score(if_raw_test, float(if_raw_train.min()), float(if_raw_train.max()))
    ae_norm = FL.normalize_ae_score(ae_raw_test, float(ae_raw_train.min()), float(ae_raw_train.max()))
    fused_score = FL.fuse_confidence_weighted(
        rf_score, if_norm, ae_norm, fusion_cfg["confidence_threshold"]
    )
    fused_pred = (fused_score > fusion_cfg["fused_decision_threshold"]).astype(int)

    return pd.DataFrame({
        "Label": test_labels,
        "y_true": y_true,
        "rf_score": rf_score, "rf_pred": rf_pred,
        "if_score": if_norm, "if_pred": if_pred,
        "ae_score": ae_norm, "ae_pred": ae_pred,
        "fused_score": fused_score, "fused_pred": fused_pred,
    })


def run_fold(slim, network_cols, labels, groups, source_row, split, scheme, fold, ae_cfg, fusion_cfg, signature):
    t0 = time.perf_counter()
    train_idx, val_idx, test_idx = split
    overlaps = {
        "test_in_train": pct_rows_sharing_groups(test_idx, train_idx, groups),
        "test_in_val": pct_rows_sharing_groups(test_idx, val_idx, groups),
        "train_in_val": pct_rows_sharing_groups(train_idx, val_idx, groups),
    }
    if scheme == "grouped":
        if any(v != 0.0 for v in overlaps.values()):
            raise RuntimeError(f"grouped split leaks groups: {overlaps}")
    train_counts = class_counts(labels, train_idx)
    test_counts = class_counts(labels, test_idx)
    held_out = held_out_classes(train_counts, test_counts)
    log(f"  rows train {len(train_idx):,}  val {len(val_idx):,}  test {len(test_idx):,}")
    log(f"  group overlap % test-train {overlaps['test_in_train']:.2f}  "
        f"test-val {overlaps['test_in_val']:.2f}  train-val {overlaps['train_in_val']:.2f}")
    if held_out:
        log(f"  classes absent from train but present in test (n_test): {held_out}")
    check_train_class_counts(train_counts)

    train_df = slim.iloc[train_idx].reset_index(drop=True)
    val_df = slim.iloc[val_idx].reset_index(drop=True)
    test_df = slim.iloc[test_idx].reset_index(drop=True)
    frames, model_cols = BUP.build_split_features(train_df, val_df, test_df, network_cols)
    n_train_after = len(frames["train"])

    torch.manual_seed(RANDOM_STATE + fold)
    np.random.seed(RANDOM_STATE + fold)
    rows = fit_and_score_members(frames, model_cols, ae_cfg, fusion_cfg)
    rows.insert(0, "source_row", source_row[test_idx])
    rows.insert(0, "row_id", test_idx)
    rows["class_in_train"] = rows["Label"].map(lambda c: train_counts.get(c, 0) > 0).to_numpy(dtype=bool)
    metrics, per_class = evaluate_rows(rows)
    for model in MODEL_NAMES:
        for label, stats in per_class[model].items():
            stats["n_train"] = train_counts.get(label, 0)

    meta = {
        "run_signature": signature,
        "scheme": scheme,
        "fold": fold,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "group_block": GROUP_BLOCK,
        "n_rows_total": len(labels),
        "n_train": len(train_idx),
        "n_train_after_smote": n_train_after,
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_test_attack": int(rows["y_true"].sum()),
        "pct_test_rows_group_in_train": overlaps["test_in_train"],
        "pct_test_rows_group_in_val": overlaps["test_in_val"],
        "pct_train_rows_group_in_val": overlaps["train_in_val"],
        "class_counts_train": train_counts,
        "class_counts_test": test_counts,
        "held_out_classes": held_out,
        "confidence_threshold": fusion_cfg["confidence_threshold"],
        "fused_decision_threshold": fusion_cfg["fused_decision_threshold"],
        "feature_count": len(model_cols),
        "seed": RANDOM_STATE + fold,
        "device": str(TUE.DEVICE),
        "seconds": time.perf_counter() - t0,
        "metrics": metrics,
        "per_class": per_class,
    }
    return meta, rows


def fmt(value) -> str:
    return "nan" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def collect_results(root: Path, signatures: dict) -> dict:
    results = {}
    for (scheme, fold), signature in sorted(signatures.items()):
        meta_path, rows_path = checkpoint_paths(root, signature, scheme, fold)
        entries = results.setdefault(scheme, [])
        if meta_path.exists() and rows_path.exists():
            entries.append((json.loads(meta_path.read_text(encoding="utf-8")),
                            pd.read_parquet(rows_path)))
    return results


def build_tables(results: dict, exclude_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows, class_rows = [], []
    for scheme, entries in results.items():
        for meta, fold_rows_df in entries:
            _, fold_per_class = evaluate_rows(fold_rows_df, exclude_ids)
            held = ";".join(f"{k}:{v}" for k, v in sorted(meta["held_out_classes"].items()))
            for model in MODEL_NAMES:
                fold_rows.append({
                    "scheme": scheme, "fold": meta["fold"], "model": model,
                    "n_test": meta["n_test"], "n_test_attack": meta["n_test_attack"],
                    **{m: meta["metrics"][model][m] for m in METRIC_NAMES},
                    "pct_test_rows_group_in_train": meta["pct_test_rows_group_in_train"],
                    "pct_test_rows_group_in_val": meta["pct_test_rows_group_in_val"],
                    "pct_train_rows_group_in_val": meta["pct_train_rows_group_in_val"],
                    "held_out_classes": held,
                })
                for label, stats in fold_per_class[model].items():
                    class_rows.append({"scheme": scheme, "fold": str(meta["fold"]), "model": model,
                                       "label": label, "n": stats["n"],
                                       "n_train": meta["per_class"][model][label]["n_train"],
                                       "n_excluded_common": stats["n"] - stats["n_seen"],
                                       "recall": stats["recall"], "recall_seen": stats["recall_seen"]})
        if entries:
            pooled = pd.concat([rows for _, rows in entries], ignore_index=True)
            _, per_class = evaluate_rows(pooled, exclude_ids)
            train_totals = {}
            for meta, _ in entries:
                for label, n in meta["class_counts_train"].items():
                    train_totals[label] = train_totals.get(label, 0) + n
            for model in MODEL_NAMES:
                for label, stats in per_class[model].items():
                    class_rows.append({"scheme": scheme, "fold": "pooled", "model": model,
                                       "label": label, "n": stats["n"],
                                       "n_train": train_totals.get(label, 0),
                                       "n_excluded_common": stats["n"] - stats["n_seen"],
                                       "recall": stats["recall"], "recall_seen": stats["recall_seen"]})
    return pd.DataFrame(fold_rows), pd.DataFrame(class_rows)


def build_summary(results: dict, fold_table: pd.DataFrame, exclude_ids, excluded_by_class: dict) -> dict:
    summary = {"schemes": {}, "common_excluded_rows": {"n": int(len(exclude_ids)), "by_class": excluded_by_class}}
    for scheme, entries in results.items():
        if not entries:
            continue
        pooled_rows = pd.concat([rows for _, rows in entries], ignore_index=True)
        pooled_metrics, _ = evaluate_rows(pooled_rows)
        seen_metrics, _ = evaluate_rows(pooled_rows, exclude_ids, seen_only=True)
        sub = fold_table[fold_table["scheme"] == scheme]
        mean, sd = {}, {}
        for model in MODEL_NAMES:
            part = sub[sub["model"] == model][list(METRIC_NAMES)]
            mean[model] = part.mean().to_dict()
            sd[model] = part.std(ddof=1).to_dict()
        by_fold = sub[sub["model"] == "rf"]
        summary["schemes"][scheme] = {
            "folds": [int(m["fold"]) for m, _ in entries],
            "n_pooled_rows": int(len(pooled_rows)),
            "n_rows_of_classes_absent_from_train": int((~pooled_rows["class_in_train"]).sum()),
            "n_rows_excluded_common": int((~seen_mask(pooled_rows, exclude_ids)).sum()),
            "n_rows_in_seen_population": int(seen_mask(pooled_rows, exclude_ids).sum()),
            "held_out_classes_by_fold": {str(m["fold"]): m["held_out_classes"] for m, _ in entries},
            "mean_across_folds": mean,
            "sd_across_folds": sd,
            "pooled": pooled_metrics,
            "pooled_excluding_absent_classes": seen_metrics,
            "n_unique_row_ids": int(pooled_rows["row_id"].nunique()),
            "row_id_set_hash": array_hash(np.sort(pooled_rows["row_id"].unique())),
            "pct_test_rows_group_in_train_mean": float(by_fold["pct_test_rows_group_in_train"].mean()),
            "pct_test_rows_group_in_val_mean": float(by_fold["pct_test_rows_group_in_val"].mean()),
            "pct_train_rows_group_in_val_mean": float(by_fold["pct_train_rows_group_in_val"].mean()),
        }
    valid, reason = comparison_status(summary["schemes"])
    summary["comparison_valid"] = valid
    summary["comparison_reason"] = reason
    if valid:
        r, g = summary["schemes"]["random"], summary["schemes"]["grouped"]
        summary["pooled_delta_grouped_minus_random"] = {
            "all_rows": {model: {m: _delta(g["pooled"][model][m], r["pooled"][model][m])
                                 for m in METRIC_NAMES} for model in MODEL_NAMES},
            "excluding_absent_classes": {
                model: {m: _delta(g["pooled_excluding_absent_classes"][model][m],
                                  r["pooled_excluding_absent_classes"][model][m])
                        for m in METRIC_NAMES} for model in MODEL_NAMES},
        }
    return clean(summary)


def comparison_status(schemes: dict) -> tuple[bool, str]:
    if not {"random", "grouped"} <= set(schemes):
        return False, f"both schemes are required; present: {sorted(schemes)}"
    r, g = schemes["random"], schemes["grouped"]
    if r["row_id_set_hash"] != g["row_id_set_hash"] or r["n_unique_row_ids"] != g["n_unique_row_ids"]:
        return False, (f"the schemes cover different row sets (random {r['n_unique_row_ids']:,} rows over "
                       f"folds {r['folds']}, grouped {g['n_unique_row_ids']:,} rows over folds {g['folds']}); "
                       "run all folds of both schemes for a like-for-like comparison")
    return True, "both schemes cover the identical row set"


def _delta(a, b):
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def _nz(value):
    return float("nan") if value is None else value


def _held_out_lines(schemes: dict) -> list[str]:
    lines = ["  classes absent from train in a fold (held-out categories, not leakage):"]
    any_held = False
    for scheme, block in schemes.items():
        for fold, held in block["held_out_classes_by_fold"].items():
            if held:
                any_held = True
                lines.append(f"    {scheme} fold {fold}: " +
                             ", ".join(f"{k} (n_test {v})" for k, v in sorted(held.items())))
    if not any_held:
        lines.append("    none")
    return lines


def comparison_text(summary: dict, class_table: pd.DataFrame) -> list[str]:
    schemes = summary["schemes"]
    if not summary["comparison_valid"]:
        lines = [f"  WARNING: no like-for-like comparison is possible: {summary['comparison_reason']}.",
                 "  No random-vs-grouped deltas are reported."]
        lines.extend(_held_out_lines(schemes))
        return lines
    r, g = schemes["random"], schemes["grouped"]
    table = []
    for model in MODEL_NAMES:
        for m in METRIC_NAMES:
            rv, gv = _nz(r["pooled"][model][m]), _nz(g["pooled"][model][m])
            rs = _nz(r["pooled_excluding_absent_classes"][model][m])
            gs = _nz(g["pooled_excluding_absent_classes"][model][m])
            table.append({"model": model, "metric": m, "random": rv, "grouped": gv,
                          "delta": gv - rv, "random_seen": rs, "grouped_seen": gs,
                          "delta_seen": gs - rs})
    lines = [pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.4f}")]
    lines.append("")
    common = summary["common_excluded_rows"]
    lines.append("  _seen columns drop the same row set from both schemes: every row whose class had 0")
    lines.append("  train rows in that row's fold under ANY scheme of this run. The leakage effect is")
    lines.append("  read from delta_seen, not delta.")
    lines.append(f"  common excluded rows {common['n']:,}  by class {common['by_class']}")
    lines.append(f"  seen population rows  random {r['n_rows_in_seen_population']:,}  "
                 f"grouped {g['n_rows_in_seen_population']:,}")
    lines.append("")
    lines.extend(_held_out_lines(schemes))
    lines.append("")
    lines.append("  per-class pooled recall; seen = rows outside the common exclusion set (rows whose class had "
                 "no train rows in that fold under any scheme)")
    pooled = class_table[class_table["fold"] == "pooled"]
    wide = pooled.pivot_table(index=["label", "model"], columns="scheme",
                              values=["recall", "recall_seen", "n", "n_excluded_common"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["delta_seen"] = wide["recall_seen_grouped"] - wide["recall_seen_random"]
    wide[["n_random", "n_grouped", "n_excluded_common_grouped"]] = wide[
        ["n_random", "n_grouped", "n_excluded_common_grouped"]].astype("Int64")
    keep = ["label", "model", "n_random", "n_grouped", "n_excluded_common_grouped", "recall_random",
            "recall_grouped", "recall_seen_random", "recall_seen_grouped", "delta_seen"]
    lines.append(wide[keep].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return lines


def write_outputs(results: dict, paths: dict[str, Path], args, run_id: str, signatures: dict) -> None:
    exclude_ids, excluded_by_class = excluded_rows(results)
    fold_table, class_table = build_tables(results, exclude_ids)
    if fold_table.empty:
        log("  no checkpoints available; nothing to aggregate")
        return
    summary = build_summary(results, fold_table, exclude_ids, excluded_by_class)
    summary["run"] = {"run_id": run_id, "schemes": list(args.schemes), "folds": list(args.folds),
                      "max_rows": args.max_rows, "outer_splits": OUTER_SPLITS,
                      "group_block": GROUP_BLOCK,
                      "signatures": {f"{s}_fold{f}": sig for (s, f), sig in sorted(signatures.items())}}
    paths["folds"].parent.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(paths["folds"], index=False)
    class_table.to_csv(paths["per_class"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    section("per-scheme results")
    for scheme, block in summary["schemes"].items():
        log(f"{scheme}: folds {block['folds']}  pooled rows {block['n_pooled_rows']:,}  "
            f"test-train {block['pct_test_rows_group_in_train_mean']:.2f}%  "
            f"test-val {block['pct_test_rows_group_in_val_mean']:.2f}%  "
            f"train-val {block['pct_train_rows_group_in_val_mean']:.2f}%  "
            f"rows of classes absent from train {block['n_rows_of_classes_absent_from_train']:,}")
        for model in MODEL_NAMES:
            mean, sd, pooled = (block["mean_across_folds"][model], block["sd_across_folds"][model],
                                block["pooled"][model])
            log(f"  {model:5s} " + "  ".join(
                f"{m} {fmt(mean[m])}+-{fmt(sd[m])} (pooled {fmt(pooled[m])})" for m in METRIC_NAMES))
    if summary["comparison_valid"]:
        section("random vs grouped, pooled over the identical row set")
    else:
        section("random vs grouped: NOT COMPARABLE (different row sets)")
    for line in comparison_text(summary, class_table):
        log(line)
    log("")
    if not summary["comparison_valid"]:
        log("  Per-scheme pooled numbers above describe different test rows and must not be differenced.")
    else:
        write_reading_notes()
    log("  Pooled AUC concatenates scores normalized per fold with that fold's train constants.")
    paths["report"].write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    for key in ("folds", "per_class", "summary", "report"):
        log(f"[saved] {paths[key]}")


def write_reading_notes() -> None:
    log("  Reading the table: delta is grouped minus random with identical training code. A gap")
    log("  is attributable to temporal-autocorrelation leakage only where the class had train rows")
    log("  in the grouped fold, so read delta_seen. Grouped CV can leave a class with no train")
    log("  rows (a class confined to one group is held out whole), which is a zero-day effect and")
    log("  not leakage, and those rows are excluded from the _seen columns and listed above.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Random vs group-aware CV robustness check")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=sorted(SCHEMES))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(OUTER_SPLITS)))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    for fold in args.folds:
        if not 0 <= fold < OUTER_SPLITS:
            parser.error(f"--folds values must be in [0, {OUTER_SPLITS - 1}]")
    args.folds = sorted(set(args.folds))
    args.schemes = [s for s in SCHEMES if s in set(args.schemes)]
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    smoke = args.max_rows is not None
    root = checkpoint_root(smoke)

    section("split robustness: random vs group-aware stratified CV")
    df, source_row = load_population(args.max_rows)
    network_cols = derive_network_cols(df)
    labels = df["Label"].to_numpy()
    groups = build_group_ids(df)
    fingerprint = data_fingerprint(df, network_cols, source_row, args.max_rows)
    slim = df[network_cols + BUP.REFERENCE_COLS]
    del df
    ae_cfg = json.loads(AE_CONFIG_PATH.read_text(encoding="utf-8"))
    fusion_cfg = json.loads(FUSION_CONFIG_PATH.read_text(encoding="utf-8"))
    log(f"  rows {len(labels):,}  groups {groups.max() + 1:,}  network features {len(network_cols)}  "
        f"device {TUE.DEVICE}  smoke {smoke}")
    log(f"  schemes {args.schemes}  folds {args.folds}  checkpoints {root}")
    log(f"  fusion confidence_threshold {fusion_cfg['confidence_threshold']}  "
        f"fused_decision_threshold {fusion_cfg['fused_decision_threshold']}")

    signatures = {}
    for scheme in args.schemes:
        outer_ids = assign_outer_folds(labels, groups, scheme)
        for fold in args.folds:
            section(f"{scheme} fold {fold}")
            split = split_fold(labels, groups, outer_ids, fold, scheme)
            signature = run_signature(scheme, fold, split[2], fingerprint, ae_cfg, fusion_cfg)
            signatures[(scheme, fold)] = signature
            if load_checkpoint(root, scheme, fold, signature, split[2]) is not None:
                log(f"  checkpoint signature {signature} matches; skipping")
                continue
            meta, rows = run_fold(slim, network_cols, labels, groups, source_row, split,
                                  scheme, fold, ae_cfg, fusion_cfg, signature)
            save_checkpoint(root, meta, rows)
            for model in MODEL_NAMES:
                m = meta["metrics"][model]
                log(f"  {model:5s} " + "  ".join(f"{k} {fmt(m[k])}" for k in METRIC_NAMES))
            log(f"  fold finished in {meta['seconds']:.1f}s")

    run_id = run_id_for(signatures.values())
    paths = output_paths(smoke, run_id)
    write_outputs(collect_results(root, signatures), paths, args, run_id, signatures)


if __name__ == "__main__":
    main()
