"""LOAO zero-day comparison of the deployed fusion rule against calibrated fusion.

Per fold the fold's TRAIN side is split into a fit set (models trained here) and a
calibration set (isotonic maps fit here). The held-out attack category never appears
in either, so the maps are fit without a single example of it.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.fusion_loao_trainnorm import (
    BEHAVIORAL_PROCESSED_DIR, BENIGN_INTERNAL_VAL_FRAC, CATEGORIES, DEVICE, IF_CONTAMINATION,
    IF_N_ESTIMATORS_LOAO, LOAO_BENIGN_HOLDOUT_SIZE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, RF_CLASS_WEIGHT,
    RF_MAX_DEPTH, RF_N_ESTIMATORS, RF_SENTINEL_COLS, ROOT, SMOTE_FLAT_CAP, SMOTE_RANDOM_STATE,
    SMOTE_TARGET_MULTIPLIER, add_missing_flags_and_impute_fold, apply_smote_loao, build_loao_split,
    checkpoint_directory, compute_reconstruction_error, compute_run_signature, load_checkpoint,
    save_checkpoint, slugify_category, train_autoencoder,
)

CHECKPOINT_DIR = REPORTS_DIR / "fusion_loao_calibrated_checkpoints"
RESULTS_CSV = REPORTS_DIR / "fusion_loao_calibrated_results.csv"
REPORT_TXT = REPORTS_DIR / "fusion_loao_calibrated_report.txt"

CALIBRATION_FRAC = 0.25
CANDIDATES = ["deployed", "cal_max", "cal_noisy_or", "cal_mean", "p_rf", "p_ae"]
CANDIDATE_LABELS = {
    "deployed": "deployed max(rf, ae_norm)",
    "cal_max": "calibrated max(p_rf, p_ae)",
    "cal_noisy_or": "calibrated noisy-OR",
    "cal_mean": "calibrated mean",
    "p_rf": "p_rf alone",
    "p_ae": "p_ae alone",
}
DEPLOYED_THRESHOLD = 0.05
FOLD_ALGORITHM_VERSION = "1"  # bump when run_fold/split/calibration semantics change; report code is excluded on purpose
TRAINNORM_PATH = ROOT / "src" / "evaluation" / "fusion_loao_trainnorm.py"


def split_fit_calibration(train_df, calibration_frac=CALIBRATION_FRAC, random_state=RANDOM_STATE):
    """Positional index split of a fold's TRAIN frame, stratified by Label."""
    positions = np.arange(len(train_df))
    fit_pos, cal_pos = train_test_split(
        positions, test_size=calibration_frac, random_state=random_state,
        stratify=train_df["Label"].to_numpy(),
    )
    return np.sort(fit_pos), np.sort(cal_pos)


def fit_isotonic(scores, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return iso


def calibrate_eval_scores(rf_eval, ae_eval, y_cal, n_cal):
    """rf_eval/ae_eval hold the calibration rows FIRST, then the zero-day test rows.

    Both maps are fit on the first n_cal entries only; the remainder is only transformed.
    """
    rf_map = fit_isotonic(rf_eval[:n_cal], y_cal)
    ae_map = fit_isotonic(ae_eval[:n_cal], y_cal)
    return rf_map.predict(rf_eval), ae_map.predict(ae_eval), rf_map, ae_map


def combine(p_rf, p_ae, how):
    p_rf = np.asarray(p_rf, dtype=np.float64)
    p_ae = np.asarray(p_ae, dtype=np.float64)
    if how == "max":
        return np.maximum(p_rf, p_ae)
    if how == "noisy_or":
        return 1.0 - (1.0 - p_rf) * (1.0 - p_ae)
    if how == "mean":
        return (p_rf + p_ae) / 2.0
    raise ValueError(how)


def candidate_scores(df):
    """Per-row candidate score columns from a scores frame."""
    return {
        "deployed": np.maximum(df["rf_prob"].to_numpy(), df["ae_norm"].to_numpy()),
        "cal_max": combine(df["p_rf"], df["p_ae"], "max"),
        "cal_noisy_or": combine(df["p_rf"], df["p_ae"], "noisy_or"),
        "cal_mean": combine(df["p_rf"], df["p_ae"], "mean"),
        "p_rf": df["p_rf"].to_numpy(),
        "p_ae": df["p_ae"].to_numpy(),
    }


def brier(prob, y):
    prob = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((prob - np.asarray(y, dtype=np.float64)) ** 2))


def run_fold(cat, joint_full_df, rf_raw_feature_cols, ae_config):
    train_df_fold, test_df_fold = build_loao_split(joint_full_df, cat)
    fit_pos, cal_pos = split_fit_calibration(train_df_fold)
    fit_df = train_df_fold.iloc[fit_pos].reset_index(drop=True)
    cal_df = train_df_fold.iloc[cal_pos].reset_index(drop=True)
    assert cat not in set(fit_df["Label"]) and cat not in set(cal_df["Label"])
    n_cal = len(cal_df)
    eval_df = pd.concat([cal_df, test_df_fold], ignore_index=True)
    y_eval = (eval_df["Label"] != "BENIGN").astype(int).to_numpy()
    y_cal = y_eval[:n_cal]

    rf_fit, rf_eval = add_missing_flags_and_impute_fold(fit_df, eval_df)
    rf_scaler = MinMaxScaler().fit(rf_fit[rf_raw_feature_cols])
    rf_fit[rf_raw_feature_cols] = rf_scaler.transform(rf_fit[rf_raw_feature_cols])
    rf_eval[rf_raw_feature_cols] = rf_scaler.transform(rf_eval[rf_raw_feature_cols])
    rf_feature_cols = rf_raw_feature_cols + [f"{c}_missing" for c in RF_SENTINEL_COLS]
    assert len(rf_feature_cols) == 80
    X_rf_fit_res, y_rf_fit_res = apply_smote_loao(
        rf_fit[rf_feature_cols].to_numpy(dtype=np.float64), rf_fit["Label"].to_numpy()
    )
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_rf_fit_res, y_rf_fit_res)
    benign_idx = list(rf_model.classes_).index("BENIGN")
    rf_prob = 1.0 - rf_model.predict_proba(rf_eval[rf_feature_cols].to_numpy(dtype=np.float64))[:, benign_idx]
    del rf_model, X_rf_fit_res, y_rf_fit_res

    ifae_fit, ifae_eval = add_missing_flags_and_impute_fold(fit_df, eval_df)
    feature_cols = [
        c for c in ifae_fit.columns
        if c not in ("Label", "orig_row_index") and pd.api.types.is_numeric_dtype(ifae_fit[c])
    ]
    assert len(feature_cols) == 92
    scale_cols = [c for c in feature_cols if c not in [f"{c}_missing" for c in RF_SENTINEL_COLS]]
    scaler = MinMaxScaler().fit(ifae_fit[scale_cols])
    ifae_fit[scale_cols] = scaler.transform(ifae_fit[scale_cols])
    ifae_eval[scale_cols] = scaler.transform(ifae_eval[scale_cols])
    X_fit = ifae_fit[feature_cols].to_numpy(dtype=np.float64)
    X_eval = ifae_eval[feature_cols].to_numpy(dtype=np.float64)

    if_model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS_LOAO, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_fit)
    if_raw = -if_model.score_samples(X_eval)

    benign_fit_df = ifae_fit[ifae_fit["Label"] == "BENIGN"]
    ae_train_df, ae_internal_df = train_test_split(
        benign_fit_df, test_size=BENIGN_INTERNAL_VAL_FRAC, random_state=RANDOM_STATE
    )
    ae_model = train_autoencoder(
        ae_train_df[feature_cols].to_numpy(dtype=np.float32),
        ae_internal_df[feature_cols].to_numpy(dtype=np.float32),
        input_dim=len(feature_cols), config=ae_config, device=DEVICE,
    )
    ae_raw_fit = compute_reconstruction_error(ae_model, X_fit.astype(np.float32), DEVICE)
    ae_raw = compute_reconstruction_error(ae_model, X_eval.astype(np.float32), DEVICE).astype(np.float64)
    ae_min, ae_max = float(ae_raw_fit.min()), float(ae_raw_fit.max())
    ae_norm = (ae_raw - ae_min) / (ae_max - ae_min)

    p_rf, p_ae, _, _ = calibrate_eval_scores(rf_prob, ae_raw, y_cal, n_cal)

    is_test = np.arange(len(eval_df)) >= n_cal
    y_test = y_eval[is_test]
    result = {
        "category": cat,
        "n_fit_rows": len(fit_df), "n_calibration_rows": n_cal, "n_test_rows": int(is_test.sum()),
        "n_fit_attack_rows": int((fit_df["Label"] != "BENIGN").sum()),
        "n_calibration_attack_rows": int(y_cal.sum()),
        "n_test_attack_rows": int(y_test.sum()),
        "n_test_benign_rows": int((y_test == 0).sum()),
        "ae_fit_min": ae_min, "ae_fit_max": ae_max,
        "brier_rf_raw_test": brier(rf_prob[is_test], y_test),
        "brier_rf_cal_test": brier(p_rf[is_test], y_test),
        "brier_ae_raw_test": brier(ae_norm[is_test], y_test),
        "brier_ae_cal_test": brier(p_ae[is_test], y_test),
        "brier_rf_raw_cal_set": brier(rf_prob[:n_cal], y_cal),
        "brier_rf_cal_cal_set": brier(p_rf[:n_cal], y_cal),
        "brier_ae_raw_cal_set": brier(ae_norm[:n_cal], y_cal),
        "brier_ae_cal_cal_set": brier(p_ae[:n_cal], y_cal),
        "attack_prevalence_cal_set": float(y_cal.mean()),
        "attack_prevalence_test": float(y_test.mean()),
        "p_ae_test_max_on_attack": float(p_ae[is_test][y_test == 1].max()) if y_test.sum() else float("nan"),
    }
    scores_df = pd.DataFrame({
        "category": cat,
        "split": np.where(is_test, "test", "cal"),
        "rf_prob": rf_prob, "ae_raw": ae_raw, "ae_norm": ae_norm, "if_raw": if_raw,
        "p_rf": p_rf, "p_ae": p_ae, "y_true": y_eval,
    })
    return result, scores_df


def load_scores(cat, signature):
    directory = checkpoint_directory(signature, CHECKPOINT_DIR)
    candidates = sorted(directory.glob(f"{slugify_category(cat)}_*_scores.parquet"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        df = pd.read_parquet(path)
        if (df["run_signature"] == signature).all() and (df["category"] == cat).all():
            return df
    raise FileNotFoundError(f"no scores checkpoint for {cat}")


def threshold_grid(scores):
    q = np.concatenate([np.linspace(0, 1, 2001), 1 - np.logspace(-6, -2, 200)])
    grid = np.unique(np.quantile(scores, q))
    return np.concatenate([[grid[0] - 1e-9], grid])


def count_matrices(per_cat, grid):
    """per_cat: {cat: (scores, y)}. Returns tp, fp, n_attack, n_benign with rule score > t."""
    tp = np.zeros((len(per_cat), len(grid)))
    fp = np.zeros_like(tp)
    n_att, n_ben = np.zeros(len(per_cat)), np.zeros(len(per_cat))
    for i, (s, y) in enumerate(per_cat.values()):
        a, b = np.sort(s[y == 1]), np.sort(s[y == 0])
        tp[i] = len(a) - np.searchsorted(a, grid, side="right")
        fp[i] = len(b) - np.searchsorted(b, grid, side="right")
        n_att[i], n_ben[i] = len(a), len(b)
    return tp, fp, n_att, n_ben


def metrics_from_counts(tp, fp, n_att, n_ben):
    tp, fp = np.asarray(tp, dtype=float), np.asarray(fp, dtype=float)
    fn = n_att - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, n_att, out=np.zeros_like(tp), where=n_att > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    fpr = np.divide(fp, n_ben, out=np.zeros_like(tp), where=n_ben > 0)
    return precision, recall, f1, fpr, fn


def evaluate_candidate(per_cat, grid):
    """Pooled oracle threshold, per-category oracle threshold, and nested (leave-one-category-out) threshold."""
    cats = list(per_cat)
    tp, fp, n_att, n_ben = count_matrices(per_cat, grid)
    rows = {}

    def f1_of(tp_, fp_, na, nb):
        return metrics_from_counts(tp_, fp_, na, nb)[2]

    pooled_tp, pooled_fp = tp.sum(0), fp.sum(0)
    pooled_f1 = f1_of(pooled_tp, pooled_fp, n_att.sum(), n_ben.sum())
    j = int(np.argmax(pooled_f1))
    p, r, f, fpr, _ = metrics_from_counts(pooled_tp[j], pooled_fp[j], n_att.sum(), n_ben.sum())
    nested_tp = nested_fp = 0.0
    nested_thr = {}
    for i, cat in enumerate(cats):
        others = [k for k in range(len(cats)) if k != i]
        jf = int(np.argmax(f1_of(tp[others].sum(0), fp[others].sum(0), n_att[others].sum(), n_ben[others].sum())))
        nested_thr[cat] = float(grid[jf])
        nested_tp += tp[i, jf]
        nested_fp += fp[i, jf]
        jo = int(np.argmax(f1_of(tp[i], fp[i], n_att[i], n_ben[i])))
        po, ro, fo, fpro, _ = metrics_from_counts(tp[i, jo], fp[i, jo], n_att[i], n_ben[i])
        pn, rn, fn_, fprn, _ = metrics_from_counts(tp[i, jf], fp[i, jf], n_att[i], n_ben[i])
        rows[cat] = {
            "oracle_threshold": float(grid[jo]), "oracle_precision": float(po), "oracle_recall": float(ro),
            "oracle_f1": float(fo), "oracle_fpr": float(fpro),
            "nested_threshold": nested_thr[cat], "nested_precision": float(pn), "nested_recall": float(rn),
            "nested_f1": float(fn_), "nested_fpr": float(fprn),
            "n_attack": int(n_att[i]), "n_benign": int(n_ben[i]),
        }
    pn, rn, fn_, fprn, _ = metrics_from_counts(nested_tp, nested_fp, n_att.sum(), n_ben.sum())
    rows["POOLED"] = {
        "oracle_threshold": float(grid[j]), "oracle_precision": float(p), "oracle_recall": float(r),
        "oracle_f1": float(f), "oracle_fpr": float(fpr),
        "nested_threshold": float("nan"), "nested_precision": float(pn), "nested_recall": float(rn),
        "nested_f1": float(fn_), "nested_fpr": float(fprn),
        "n_attack": int(n_att.sum()), "n_benign": int(n_ben.sum()),
    }
    return rows


def fixed_threshold_metrics(per_cat, threshold):
    tp = fp = na = nb = 0
    for s, y in per_cat.values():
        tp += int(((s > threshold) & (y == 1)).sum())
        fp += int(((s > threshold) & (y == 0)).sum())
        na += int((y == 1).sum())
        nb += int((y == 0).sum())
    p, r, f, fpr, _ = metrics_from_counts(tp, fp, na, nb)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "fpr": float(fpr), "tp": tp, "fp": fp,
            "n_attack": na, "n_benign": nb}


def analyze(results, scores_by_cat):
    test = {c: s[s["split"] == "test"].reset_index(drop=True) for c, s in scores_by_cat.items()}
    cand = {c: candidate_scores(df) for c, df in test.items()}
    y = {c: df["y_true"].to_numpy() for c, df in test.items()}
    table = []
    evaluations = {}
    for name in CANDIDATES:
        per_cat = {c: (cand[c][name], y[c]) for c in CATEGORIES}
        grid = threshold_grid(np.concatenate([v[0] for v in per_cat.values()]))
        evaluations[name] = evaluate_candidate(per_cat, grid)
        if name == "deployed":
            evaluations["deployed_fixed_0.05"] = fixed_threshold_metrics(per_cat, DEPLOYED_THRESHOLD)
    by_cat = {r["category"]: r for r in results}
    pooled_brier = {}
    for key in ("rf_raw", "rf_cal", "ae_raw", "ae_cal"):
        col = {"rf_raw": "rf_prob", "rf_cal": "p_rf", "ae_raw": "ae_norm", "ae_cal": "p_ae"}[key]
        allv = np.concatenate([test[c][col].to_numpy() for c in CATEGORIES])
        ally = np.concatenate([y[c] for c in CATEGORIES])
        pooled_brier[key] = brier(allv, ally)
    for name in CANDIDATES:
        for cat, row in evaluations[name].items():
            rec = {"category": cat, "candidate": name, **row}
            src = by_cat.get(cat)
            for k in ("rf_raw", "rf_cal", "ae_raw", "ae_cal"):
                rec[f"brier_{k}_test"] = src[f"brier_{k}_test"] if src else pooled_brier[k]
            table.append(rec)
    return pd.DataFrame(table), evaluations, pooled_brier


def _fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_report(path, results, table, evaluations, pooled_brier, run_signature):
    rdf = pd.DataFrame(results).set_index("category").loc[CATEGORIES]
    L = []
    bar = "=" * 70
    L += [bar, "CALIBRATED FUSION vs DEPLOYED RULE -- LOAO ZERO-DAY", bar,
          f"Script: src/evaluation/fusion_loao_calibrated.py   Run signature: {run_signature[:16]}",
          "Question: does mapping each member's score to P(attack | score) with isotonic regression,",
          "then combining by probability algebra, beat the deployed max(rf_attack_probability, ae_norm)",
          "on held-out attack categories? In-dataset cannot decide this (RF saturates at F1 0.9965).", ""]

    L += [bar, "1. FIT / CALIBRATION SPLIT", bar,
          f"Each fold's TRAIN side (all BENIGN except the 20% zero-day BENIGN slice, plus every attack",
          f"category except the held-out one) is split {int((1 - CALIBRATION_FRAC) * 100)}/{int(CALIBRATION_FRAC * 100)} "
          f"stratified by Label (random_state={RANDOM_STATE}).",
          "Fit set: RF (80 features, capped SMOTE), IF-v2 and AE-v2 (92 features, BENIGN-only for AE) are",
          "trained here, with scalers/imputation/AE min-max fit here too.",
          "Calibration set: scored by the trained members, used ONLY to fit the two isotonic maps",
          "(RF attack probability -> P(attack); AE-v2 raw reconstruction error -> P(attack)).",
          "25% keeps the rarest training classes represented in both halves under stratification while",
          "leaving 75% for fitting; it is a judgment call, not tuned.", "",
          f"{'category':26s}{'fit':>10s}{'cal':>10s}{'cal atk':>10s}{'test':>10s}{'test atk':>10s}"]
    for cat, r in rdf.iterrows():
        L.append(f"{cat:26s}{int(r.n_fit_rows):>10,}{int(r.n_calibration_rows):>10,}"
                 f"{int(r.n_calibration_attack_rows):>10,}{int(r.n_test_rows):>10,}{int(r.n_test_attack_rows):>10,}")
    L += ["", "Zero-day test BENIGN slice is identical across folds (fixed seed), so pooled rows repeat it 10 times;",
          "pooled FPR is therefore the per-fold BENIGN FPR, and pooled precision is not a deployment precision.", ""]

    L += [bar, "2. LEAKAGE ARGUMENT", bar,
          "The held-out category is removed before the fit/calibration split, so neither the models nor the",
          "isotonic maps ever see one example of it. The maps are fit on calibration rows only: the function",
          "that fits them receives the calibration slice and nothing else, and tests/test_fusion_loao_calibrated.py",
          "asserts that changing every zero-day test row leaves the fitted maps bit-identical. Fitting a map on",
          "the test fold would let it learn exactly where that category's scores sit, which is the thing a",
          "zero-day fold measures. The cost is real: the map is calibrated to the other nine attack mixes and",
          "may be wrong for the held-out one. Sections 5 and 6 report how wrong rather than engineering around it.",
          "A second, smaller shift is also left in: calibration-set attack prevalence differs from the test fold's,",
          "and no prior correction was applied.", ""]

    def row(cat, cand):
        return table[(table.category == cat) & (table.candidate == cand)].iloc[0]

    L += [bar, "3. POOLED RESULTS (all 10 folds concatenated)", bar,
          "oracle = one threshold swept on the pooled zero-day scores (optimistic, picks the best cutoff using test labels).",
          "nested = per-category threshold chosen on the other nine categories, applied to the held-out one (honest).",
          "Each candidate is evaluated at its own swept threshold, never at another candidate's.", "",
          f"{'candidate':30s}{'thr(oracle)':>12s}{'P':>8s}{'R':>8s}{'F1':>8s}{'FPR':>8s}   {'nested P':>9s}{'R':>8s}{'F1':>8s}{'FPR':>8s}"]
    for n in CANDIDATES:
        p = row("POOLED", n)
        L.append(f"{CANDIDATE_LABELS[n]:30s}{_fmt(p.oracle_threshold):>12s}{_fmt(p.oracle_precision):>8s}"
                 f"{_fmt(p.oracle_recall):>8s}{_fmt(p.oracle_f1):>8s}{_fmt(p.oracle_fpr):>8s}   "
                 f"{_fmt(p.nested_precision):>9s}{_fmt(p.nested_recall):>8s}{_fmt(p.nested_f1):>8s}{_fmt(p.nested_fpr):>8s}")
    fx = evaluations["deployed_fixed_0.05"]
    L += ["", f"Deployed rule at its shipped fixed threshold 0.05: P {fx['precision']:.4f} R {fx['recall']:.4f} "
              f"F1 {fx['f1']:.4f} FPR {fx['fpr']:.4f} (tp={fx['tp']:,} of {fx['n_attack']:,} attacks, fp={fx['fp']:,} of {fx['n_benign']:,} benign).",
          "Caveat: ae_norm here is normalized on the 75% fit set and the RF saw 75% of train, so this is a",
          "re-trained baseline, not the earlier run's numbers. The earlier nested F1 for OR: max(RF, AE) was 0.9253.", ""]
    pooled = {n: row("POOLED", n) for n in CANDIDATES}
    dep, nor = pooled["deployed"], pooled["cal_noisy_or"]
    best_cal = max(("cal_max", "cal_noisy_or", "cal_mean"), key=lambda n: pooled[n].nested_f1)
    d_nested = pooled[best_cal].nested_f1 - dep.nested_f1
    d_oracle = max(pooled[n].oracle_f1 for n in ("cal_max", "cal_noisy_or", "cal_mean")) - dep.oracle_f1
    verdict = ("calibrated fusion BEAT" if d_nested > 0.005 else
               "calibrated fusion LOST TO" if d_nested < -0.005 else "calibrated fusion is within 0.005 F1 of")
    L += [f"Headline: best calibrated combiner by nested F1 is {CANDIDATE_LABELS[best_cal]} "
          f"(nested F1 {pooled[best_cal].nested_f1:.4f} vs deployed {dep.nested_f1:.4f}, delta {d_nested:+.4f}; "
          f"oracle-F1 delta {d_oracle:+.4f}). On this run {verdict} the deployed rule on pooled zero-day.", ""]

    L += [bar, "4. PER-CATEGORY RECALL (nested thresholds) AND FPR", bar,
          f"{'category':26s}{'n_atk':>8s}  {'deployed R':>10s}{'FPR':>8s}  {'noisy-OR R':>10s}{'FPR':>8s}  {'p_rf R':>8s}  {'p_ae R':>8s}   oracle-F1 dep / nOR"]
    for cat in CATEGORIES:
        d, o, a, b = row(cat, "deployed"), row(cat, "cal_noisy_or"), row(cat, "p_rf"), row(cat, "p_ae")
        L.append(f"{cat:26s}{int(d.n_attack):>8,}  {_fmt(d.nested_recall):>10s}{_fmt(d.nested_fpr):>8s}  "
                 f"{_fmt(o.nested_recall):>10s}{_fmt(o.nested_fpr):>8s}  {_fmt(a.nested_recall):>8s}  "
                 f"{_fmt(b.nested_recall):>8s}   {_fmt(d.oracle_f1)} / {_fmt(o.oracle_f1)}")
    wins = sum(row(c, "cal_noisy_or").nested_f1 > row(c, "deployed").nested_f1 for c in CATEGORIES)
    L += ["", f"Noisy-OR beats deployed on nested per-category F1 in {wins}/10 categories.",
          "Per-category F1 on a single fold mixes a tiny attack count with a 20% BENIGN slice; read recall and FPR, not F1, per row.", ""]

    L += [bar, "5. DID CALIBRATION TRANSFER TO UNSEEN CATEGORIES? (Brier on the zero-day test fold)", bar,
          "Lower is better. 'raw' is the uncalibrated score read as a probability (RF attack probability;",
          "AE ae_norm clipped to [0,1]); 'cal' is after the isotonic map fit WITHOUT the held-out category.", "",
          f"{'category':26s}{'RF raw':>9s}{'RF cal':>9s}{'AE raw':>9s}{'AE cal':>9s}   {'AE gain':>8s}  prevalence cal/test"]
    for cat, r in rdf.iterrows():
        gain = r.brier_ae_raw_test / r.brier_ae_cal_test if r.brier_ae_cal_test > 0 else float("nan")
        L.append(f"{cat:26s}{r.brier_rf_raw_test:>9.4f}{r.brier_rf_cal_test:>9.4f}{r.brier_ae_raw_test:>9.4f}"
                 f"{r.brier_ae_cal_test:>9.4f}   {gain:>7.2f}x  {r.attack_prevalence_cal_set:.3f}/{r.attack_prevalence_test:.3f}")
    pb = pooled_brier
    L += ["", f"Pooled: RF raw {pb['rf_raw']:.4f} -> cal {pb['rf_cal']:.4f}; AE raw {pb['ae_raw']:.4f} -> cal {pb['ae_cal']:.4f}.",
          "For contrast, Brier on the calibration set itself (in-sample for the maps, so optimistic):"]
    for cat, r in rdf.iterrows():
        L.append(f"  {cat:26s} RF {r.brier_rf_raw_cal_set:.4f}->{r.brier_rf_cal_cal_set:.4f}   "
                 f"AE {r.brier_ae_raw_cal_set:.4f}->{r.brier_ae_cal_cal_set:.4f}")
    improved_ae = int((rdf.brier_ae_cal_test < rdf.brier_ae_raw_test).sum())
    improved_rf = int((rdf.brier_rf_cal_test < rdf.brier_rf_raw_test).sum())
    L += ["", f"AE Brier improved in {improved_ae}/10 folds, RF Brier improved in {improved_rf}/10 folds. "
              "Brier mixes calibration with discrimination and with the prevalence shift between the calibration "
              "and test folds, so an improvement here is evidence of transfer but not a clean isolation of it.", ""]

    L += [bar, "6. WHAT THIS RUN DOES NOT SETTLE", bar,
          "- One seed, one fold structure. No confidence interval; the pooled deltas are point estimates. A",
          "  difference of a few thousandths of F1 should not be read as a ranking.",
          "- The oracle columns choose a threshold with test labels and flatter every candidate; only the nested",
          "  columns approximate deployment, and nested thresholds themselves come from nine categories of data.",
          "- Calibrated thresholds are posteriors only for the calibration-set prevalence, not for production",
          "  traffic. A 'P(attack) > t' reading holds in a deployment only if that prevalence is reproduced or reweighted.",
          "- Calibration was fit without the held-out category by design. A deployed system would calibrate on all",
          "  known categories and still meet unseen ones, which is this setup; but it says nothing about whether a",
          "  larger or more diverse calibration set would transfer better.",
          "- Isotonic regression is piecewise constant, so calibrated scores have large ties; thresholds are",
          "  therefore coarse and recall can jump. Platt or a smoothed map were not tried.",
          "- The synthetic overlay and MTTD are not touched. Real-traffic behaviour is untested.",
          "- Models are re-trained on 75% of each fold's train, so the deployed-rule baseline here is not",
          "  numerically the earlier run's; it is the fair same-data comparison, not a replacement for it.",
          "- Nothing here changes the shipped fusion rule or models/fusion_config.json.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(report_only=False):
    print(f"ROOT: {ROOT}  |  DEVICE: {DEVICE}")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    ae_config_path = MODELS_DIR / "autoencoder_behavioral_v2_config.json"
    with open(ae_config_path, encoding="utf-8") as f:
        ae_config = json.load(f)
    input_paths = [BEHAVIORAL_PROCESSED_DIR / f"cicids2017_behavioral_{s}.parquet" for s in ("train", "val", "test")]
    settings = {
        "random_state": RANDOM_STATE, "categories": CATEGORIES, "rf_sentinel_cols": RF_SENTINEL_COLS,
        "loao_benign_holdout_size": LOAO_BENIGN_HOLDOUT_SIZE, "calibration_frac": CALIBRATION_FRAC,
        "rf_n_estimators": RF_N_ESTIMATORS, "rf_class_weight": RF_CLASS_WEIGHT,
        "smote_flat_cap": SMOTE_FLAT_CAP, "smote_target_multiplier": SMOTE_TARGET_MULTIPLIER,
        "smote_random_state": SMOTE_RANDOM_STATE, "if_contamination": IF_CONTAMINATION,
        "if_n_estimators": IF_N_ESTIMATORS_LOAO, "benign_internal_val_frac": BENIGN_INTERNAL_VAL_FRAC,
        "device_type": DEVICE.type, "fold_algorithm_version": FOLD_ALGORITHM_VERSION,
    }
    signature = compute_run_signature(input_paths, ae_config_path, settings, code_path=Path(TRAINNORM_PATH),
                                      dependency_lock_path=ROOT / "uv.lock")
    checkpoint_directory(signature, CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Run signature: {signature}\nCheckpoints: {checkpoint_directory(signature, CHECKPOINT_DIR)}", flush=True)

    if not report_only:
        joint = pd.concat([pd.read_parquet(p) for p in input_paths], ignore_index=True)
        assert joint.shape == (2_026_398, 92), joint.shape
        rf_raw_cols = [c for c in joint.columns
                       if c not in ("Label", "orig_row_index") and not c.endswith(("_1m", "_3m", "_5m"))]
        assert len(rf_raw_cols) == 78
        t0 = time.perf_counter()
        for i, cat in enumerate(CATEGORIES, 1):
            if load_checkpoint(cat, signature, CHECKPOINT_DIR) is not None:
                print(f"[{i}/10] {cat:26s} SKIPPED (checkpoint)", flush=True)
                continue
            t = time.perf_counter()
            result, scores_df = run_fold(cat, joint, rf_raw_cols, ae_config)
            save_checkpoint(result, scores_df, signature, CHECKPOINT_DIR)
            print(f"[{i}/10] {cat:26s} fit={result['n_fit_rows']:,} cal={result['n_calibration_rows']:,} "
                  f"test={result['n_test_rows']:,}  Brier AE {result['brier_ae_raw_test']:.4f}->{result['brier_ae_cal_test']:.4f} "
                  f"RF {result['brier_rf_raw_test']:.4f}->{result['brier_rf_cal_test']:.4f}  ({time.perf_counter() - t:.0f}s)",
                  flush=True)
        print(f"Loop wall-clock: {time.perf_counter() - t0:.0f}s")

    results = [load_checkpoint(c, signature, CHECKPOINT_DIR) for c in CATEGORIES]
    assert all(r is not None for r in results), "missing checkpoint"
    scores_by_cat = {c: load_scores(c, signature) for c in CATEGORIES}
    table, evaluations, pooled_brier = analyze(results, scores_by_cat)
    table.to_csv(RESULTS_CSV, index=False)
    write_report(REPORT_TXT, results, table, evaluations, pooled_brier, signature)
    print("Saved:", RESULTS_CSV, REPORT_TXT)


if __name__ == "__main__":
    import sys
    main(report_only="--report-only" in sys.argv)
