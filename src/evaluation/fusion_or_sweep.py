"""
fusion_or_sweep.py

Test a decision-level OR combination of RF + AE-v2 against the existing cascade
(`confidence_weighted`) and the solo baselines, on the TRAIN-FOLD-NORMALIZED LOAO
scores from `src/evaluation/fusion_loao_trainnorm.py`.

Motivation: the nested sweep in `fusion_loao_trainnorm_sweep.py` found that
(a) AE-v2 solo beats the cascade on pooled F1, and (b) the cascade's own
`confidence_threshold` tuner selects 1.00 in 9/10 nested folds -- i.e. it switches the
RF-primary design off and falls back to the unsupervised mean almost always. At the
same time the per-category table shows a clean division of labour: RF owns
FTP-Patator / Web Attack - Brute Force / DoS slowloris, AE-v2 owns PortScan / DDoS.
A cascade can only ever pick one of them per row; an OR fires if EITHER is confident,
so it is the natural way to keep both specialists.

Candidates compared (every one at its own fairly-swept threshold):
  * RF solo                  -- rf_norm > t
  * AE-v2 solo               -- ae_norm > t
  * fusion: confidence_weighted -- the deployed cascade (2D: confidence + decision)
  * OR: max(RF, AE)          -- score-level OR, one shared threshold
  * OR: RF>t1 or AE>t2       -- decision-level OR, two INDEPENDENT thresholds (2D)
  * AND: RF>t1 and AE>t2     -- included only as the contrast case for the OR claim

Selection protocols, same as the sibling sweep script:
  * tuned-on-all -- thresholds picked on all 10 categories, scored on the same. Optimistic.
  * nested       -- thresholds picked on the OTHER nine, applied to the held-out one.
                    This is the number to cite.

Run from the repo root:  python -m src.evaluation.fusion_or_sweep
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DEFAULT_CHECKPOINT_DIR = REPORTS_DIR / "fusion_layer_loao_trainnorm_checkpoints"

CATEGORIES = [
    "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
    "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Attack - Brute Force",
]

# 1D grid, shared by the solo candidates and the score-level OR
GRID_1D = np.concatenate([
    np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]),
    np.linspace(0.05, 0.95, 19),
])
# coarser per-axis grid for the 2D candidates (26^2 = 676 combos per selection)
GRID_2D = np.concatenate([
    np.array([0.005, 0.01, 0.02, 0.025, 0.03, 0.04]),
    np.linspace(0.05, 0.95, 20),
])
CONF_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 1.0]


def prf1(pred, y):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return p, r, f1, fpr


# ---------------------------------------------------------------------------
# candidates: each is (kind, predict_fn); params are what the tuner searches over
# ---------------------------------------------------------------------------
def pred_rf(d, p):
    return (d["rf"] > p[0]).astype(int)


def pred_ae(d, p):
    return (d["ae"] > p[0]).astype(int)


def pred_if(d, p):
    return (d["if_"] > p[0]).astype(int)


def pred_maxor(d, p):
    return (np.maximum(d["rf"], d["ae"]) > p[0]).astype(int)


def pred_or2(d, p):
    return ((d["rf"] > p[0]) | (d["ae"] > p[1])).astype(int)


def pred_and2(d, p):
    return ((d["rf"] > p[0]) & (d["ae"] > p[1])).astype(int)


def pred_cascade(d, p):
    conf = np.abs(d["rf"] - 0.5) * 2
    score = np.where(conf >= p[0], d["rf"], (d["if_"] + d["ae"]) / 2)
    return (score > p[1]).astype(int)


CANDIDATES = {
    "RF solo":                     (pred_rf,      [GRID_1D]),
    "AE-v2 solo":                  (pred_ae,      [GRID_1D]),
    "IF-v2 solo":                  (pred_if,      [GRID_1D]),
    "fusion: confidence_weighted": (pred_cascade, [CONF_GRID, GRID_1D]),
    "OR: max(RF, AE)":             (pred_maxor,   [GRID_1D]),
    "OR: RF or AE (2 thresholds)": (pred_or2,     [GRID_2D, GRID_2D]),
    "AND: RF and AE":              (pred_and2,    [GRID_2D, GRID_2D]),
}


def best_params(d, fn, grids):
    """Exhaustive search over the candidate's own grid(s), maximizing F1."""
    y = d["y"]
    best = (-1.0, None)
    if len(grids) == 1:
        for a in grids[0]:
            f1 = prf1(fn(d, (a,)), y)[2]
            if f1 > best[0]:
                best = (f1, (float(a),))
    else:
        for a in grids[0]:
            for b in grids[1]:
                f1 = prf1(fn(d, (a, b)), y)[2]
                if f1 > best[0]:
                    best = (f1, (float(a), float(b)))
    return best[1]


def load_scores(CHECKPOINT_DIR):
    if not CHECKPOINT_DIR.exists():
        raise SystemExit(f"no checkpoint dir: {CHECKPOINT_DIR}")
    for dirpath in sorted((p for p in CHECKPOINT_DIR.iterdir() if p.is_dir()),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        frames = {}
        for path in sorted(dirpath.glob("*_scores.parquet")):
            df = pd.read_parquet(path)
            cat = df["category"].iloc[0]
            if cat not in frames or path.stat().st_mtime > frames[cat][1]:
                frames[cat] = (df, path.stat().st_mtime)
        if set(frames) >= set(CATEGORIES):
            pooled = pd.concat([frames[c][0] for c in CATEGORIES], ignore_index=True)
            print(f"Using signature dir: {dirpath.name}")
            print(f"Pooled rows: {len(pooled):,}  attack rows: {int(pooled['y_true'].sum()):,}")
            return pooled, dirpath.name
    raise SystemExit("no signature dir has all 10 category score files")


def subset(pooled, mask):
    return {
        "rf": pooled["rf_norm"].to_numpy()[mask],
        "if_": pooled["if_norm"].to_numpy()[mask],
        "ae": pooled["ae_norm"].to_numpy()[mask],
        "y": pooled["y_true"].to_numpy()[mask],
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR),
                    help="directory of per-fold *_scores.parquet checkpoints; a bare name is "
                         "resolved under reports/")
    ap.add_argument("--out-prefix", default=None,
                    help="basename for the output CSV/JSON; defaults to the checkpoint "
                         "directory name with _or_comparison/_or_summary appended")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = REPORTS_DIR / args.checkpoint_dir
    prefix = args.out_prefix or ckpt.name.replace("_checkpoints", "")
    OUT_CSV = REPORTS_DIR / f"{prefix}_or_comparison.csv"
    OUT_JSON = REPORTS_DIR / f"{prefix}_or_summary.json"
    pooled, signature = load_scores(ckpt)
    cat_arr = pooled["category"].to_numpy()
    all_mask = np.ones(len(pooled), dtype=bool)
    d_all = subset(pooled, all_mask)

    pooled_rows, percat_rows = [], []

    for name, (fn, grids) in CANDIDATES.items():
        # ---------- tuned-on-all ----------
        p_all = best_params(d_all, fn, grids)
        pred_all = fn(d_all, p_all)
        p, r, f1, fpr = prf1(pred_all, d_all["y"])
        pooled_rows.append({"candidate": name, "selection": "tuned-on-all",
                            "params": "|".join(f"{v:g}" for v in p_all),
                            "precision": p, "recall": r, "f1": f1, "fpr": fpr})

        # ---------- nested ----------
        pred_nested = np.zeros(len(pooled), dtype=int)
        for cat in CATEGORIES:
            held = cat_arr == cat
            p_sel = best_params(subset(pooled, ~held), fn, grids)
            d_held = subset(pooled, held)
            pr = fn(d_held, p_sel)
            pred_nested[held] = pr
            cp, cr, cf1, cfpr = prf1(pr, d_held["y"])
            percat_rows.append({"candidate": name, "selection": "nested", "category": cat,
                                "selected_params": "|".join(f"{v:g}" for v in p_sel),
                                "n_attack_rows": int(d_held["y"].sum()),
                                "recall": cr, "fpr": cfpr, "precision": cp, "f1": cf1})
        p, r, f1, fpr = prf1(pred_nested, d_all["y"])
        pooled_rows.append({"candidate": name, "selection": "nested", "params": "per-fold",
                            "precision": p, "recall": r, "f1": f1, "fpr": fpr})
        print(f"  {name:32s} nested F1 = {f1:.4f}   recall {r:.4f}  fpr {fpr:.4f}")

    pooled_df = pd.DataFrame(pooled_rows)
    percat_df = pd.DataFrame(percat_rows)

    header = [
        "# RF+AE decision-level OR vs. the deployed cascade and the solo baselines.",
        "# Scores: TRAIN-FOLD-NORMALIZED LOAO checkpoints, read back without retraining.",
        f"# Source checkpoint directory: {ckpt}",
        "# IF-v2 is reported solo and used inside the cascade's fallback, but is not an OR",
        "# member: its LOAO FPR is roughly 10x the other two, so ORing it in buys recall at a",
        "# false-positive cost a SOC cannot absorb.",
        f"# Source checkpoint signature: {signature}",
        f"# Pooled rows: {len(pooled)}  attack rows: {int(pooled['y_true'].sum())}",
        "# 'nested' = thresholds selected on the other 9 categories; cite these, not tuned-on-all.",
        "#",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        f.write("# --- pooled ---\n")
        pooled_df.sort_values(["selection", "f1"], ascending=[True, False]).to_csv(f, index=False)
        f.write("#\n# --- per held-out category (nested selection) ---\n")
        percat_df.to_csv(f, index=False)
    print("\nSaved:", OUT_CSV)

    nested = pooled_df[pooled_df["selection"] == "nested"].sort_values("f1", ascending=False)
    print("\n=== pooled, nested selection ===")
    print(nested[["candidate", "precision", "recall", "f1", "fpr"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== per-category recall, nested ===")
    piv = percat_df.pivot(index="category", columns="candidate", values="recall")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "source_signature": signature,
            "normalization": "train_fold_fitted",
            "pooled_rows": int(len(pooled)),
            "nested_pooled_f1": dict(zip(nested["candidate"], nested["f1"])),
            "nested_pooled_recall": dict(zip(nested["candidate"], nested["recall"])),
            "nested_pooled_fpr": dict(zip(nested["candidate"], nested["fpr"])),
        }, f, indent=2)
    print("Saved:", OUT_JSON)


if __name__ == "__main__":
    main()
