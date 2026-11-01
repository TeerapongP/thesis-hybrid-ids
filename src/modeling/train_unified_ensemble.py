"""Train all three ensemble members on the single unified population.

Why this replaces the notebooks. `02_random_forest.ipynb` reads the MachineLearningCSV
matrix and `06_behavioral_v2_retrain.ipynb` reads the behavioral one, which is exactly the
split the advisor objected to: the members were never on the same dataset, and the
row-hash join that reconciled them kept only 80.36% of the test split. `build_unified_population.py`
removed that split. This script is the other half: one source, one feature set, three models.

Each model is saved as soon as it finishes, so a crash costs one model rather than the run.

Run from the repo root with the Anaconda interpreter:
    python -m src.modeling.train_unified_ensemble
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features/unified"
MODELS = ROOT / "models"
REPORT = ROOT / "reports/train_unified_ensemble_report.txt"
METRICS = ROOT / "reports/unified_val_metrics.csv"
NORM = ROOT / "reports/unified_norm_constants.json"

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NON_FEATURE_COLS = [
    "Label", "technique_id", "technique_name", "tactic", "mapping_confidence",
    "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique",
]

RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = "balanced"
IF_CONTAMINATION = 0.20
AE_PERCENTILE = 99
AE_BENIGN_VAL_FRAC = 0.10

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 70)
    log(title)
    log("=" * 70)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super().__init__()
        enc_dims = [input_dim] + list(hidden_dims) + [bottleneck_dim]
        dec_dims = [bottleneck_dim] + list(hidden_dims)[::-1] + [input_dim]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, X, batch=200_000):
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(DEVICE)
            out[i:i + len(xb)] = torch.mean((model(xb) - xb) ** 2, dim=1).cpu().numpy()
    return out


def train_autoencoder(X_fit, X_val, input_dim, cfg):
    model = Autoencoder(input_dim, cfg["hidden_dims"], cfg["bottleneck_dim"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    crit = nn.MSELoss()
    fit_t = torch.tensor(X_fit, dtype=torch.float32).to(DEVICE)
    val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    best, best_state, patience = float("inf"), None, 0
    epochs_run = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(fit_t), device=DEVICE)
        for s in range(0, len(fit_t), cfg["batch_size"]):
            batch = fit_t[perm[s:s + cfg["batch_size"]]]
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(val_t), val_t).item()
        epochs_run = epoch + 1
        if vl < best - 1e-6:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    return model, epochs_run


def binary_metrics(y_true, pred, score):
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, score),
    }


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    section("STEP 0: load the unified population")
    train = pd.read_parquet(FEAT / "cicids2017_unified_train.parquet")
    val = pd.read_parquet(FEAT / "cicids2017_unified_val.parquet")
    feature_cols = [c for c in train.columns
                    if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(train[c])]
    assert feature_cols == [c for c in val.columns
                            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(val[c])], \
        "train and val feature columns differ"
    log(f"  train {len(train):,}  val {len(val):,}  features {len(feature_cols)}  device {DEVICE}")
    log("  all three members read these same columns; no join, no second feature set")

    X_train = train[feature_cols].to_numpy(dtype=np.float64)
    X_val = val[feature_cols].to_numpy(dtype=np.float64)
    y_train_label = train["Label"].to_numpy()
    y_val = (val["Label"] != "BENIGN").astype(int).to_numpy()
    rows = []

    # ── Random Forest ────────────────────────────────────────────────
    section("STEP 1: Random Forest (multi-class, supervised)")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight=RF_CLASS_WEIGHT,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train_label)
    log(f"  fitted in {time.perf_counter() - t0:.1f}s")
    joblib.dump(rf, MODELS / "random_forest_unified.pkl")
    log(f"  saved models/random_forest_unified.pkl")

    proba = rf.predict_proba(X_val)
    benign_idx = list(rf.classes_).index("BENIGN")
    rf_score = 1.0 - proba[:, benign_idx]
    rf_pred = (rf.classes_[proba.argmax(1)] != "BENIGN").astype(int)
    m = binary_metrics(y_val, rf_pred, rf_score)
    rows.append({"model": "random_forest", **m})
    log(f"  val  P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")
    rf_train_score = 1.0 - rf.predict_proba(X_train)[:, benign_idx]

    # ── Isolation Forest ─────────────────────────────────────────────
    section("STEP 2: Isolation Forest (unsupervised)")
    t0 = time.perf_counter()
    iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)
    log(f"  fitted in {time.perf_counter() - t0:.1f}s")
    joblib.dump(iso, MODELS / "isolation_forest_unified.pkl")
    log("  saved models/isolation_forest_unified.pkl")

    if_train = -iso.score_samples(X_train)
    if_val = -iso.score_samples(X_val)
    if_pred = (iso.predict(X_val) == -1).astype(int)
    m = binary_metrics(y_val, if_pred, if_val)
    rows.append({"model": "isolation_forest", **m})
    log(f"  val  P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")

    # ── Autoencoder ──────────────────────────────────────────────────
    section("STEP 3: Autoencoder (unsupervised, BENIGN rows only)")
    ae_cfg = json.loads((MODELS / "autoencoder_behavioral_v2_config.json").read_text())
    benign_mask = train["Label"].to_numpy() == "BENIGN"
    X_benign = X_train[benign_mask].astype(np.float32)
    log(f"  BENIGN training rows: {len(X_benign):,}")
    fit_X, val_X = train_test_split(X_benign, test_size=AE_BENIGN_VAL_FRAC, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    ae, epochs_run = train_autoencoder(fit_X, val_X, len(feature_cols), ae_cfg)
    log(f"  trained {epochs_run} epochs in {time.perf_counter() - t0:.1f}s")
    torch.save(ae.state_dict(), MODELS / "autoencoder_unified.pt")

    ae_internal = reconstruction_error(ae, val_X)
    threshold = float(np.percentile(ae_internal, AE_PERCENTILE))
    ae_train = reconstruction_error(ae, X_train.astype(np.float32))
    ae_val = reconstruction_error(ae, X_val.astype(np.float32))
    ae_pred = (ae_val > threshold).astype(int)
    m = binary_metrics(y_val, ae_pred, ae_val)
    rows.append({"model": "autoencoder", **m})
    log(f"  threshold (p{AE_PERCENTILE} of internal-val BENIGN error): {threshold:.8f}")
    log(f"  val  P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")

    cfg_out = dict(ae_cfg)
    # The behavioral-v2 config is reused for the ARCHITECTURE only. Its population
    # provenance describes the row-hash-joined population this run replaces, so carrying
    # those keys forward would state a BENIGN shortfall that no longer exists: the unified
    # population has MORE benign rows than the original pipeline, not 23.57% fewer.
    for stale in ("baseline_train_benign_row_count", "benign_row_count_reduction_pct"):
        cfg_out.pop(stale, None)
    cfg_out.update({"input_dim": len(feature_cols), "chosen_threshold": threshold,
                    "epochs_run": epochs_run, "train_benign_row_count": int(len(X_benign)),
                    "architecture_inherited_from": "autoencoder_behavioral_v2_config.json",
                    "population": "unified (100% coverage, no row-hash join)"})
    (MODELS / "autoencoder_unified_config.json").write_text(json.dumps(cfg_out, indent=2))
    log("  saved models/autoencoder_unified.pt and autoencoder_unified_config.json")

    # ── normalization constants, fit on TRAIN ────────────────────────
    section("STEP 4: train-fitted normalization constants for the fusion layer")
    norm = {
        "if_min": float(if_train.min()), "if_max": float(if_train.max()),
        "ae_min": float(ae_train.min()), "ae_max": float(ae_train.max()),
        "fit_population": str((FEAT / "cicids2017_unified_train.parquet").relative_to(ROOT)).replace("\\", "/"),
        "fit_rows": int(len(X_train)),
        "feature_count": len(feature_cols),
    }
    NORM.write_text(json.dumps(norm, indent=2))
    for k in ("if_min", "if_max", "ae_min", "ae_max"):
        log(f"  {k}: {norm[k]:.10f}")
    log("  fit on TRAIN, never val or test: the LOAO-selected threshold only transfers on this basis")

    # ── fused rule at the currently deployed threshold ───────────────
    section("STEP 5: deployed rf_ae_or rule on val, for reference only")
    ae_norm = (ae_val - norm["ae_min"]) / (norm["ae_max"] - norm["ae_min"])
    fused = np.maximum(rf_score, ae_norm)
    for thr in (0.05, 0.27, 0.50):
        m = binary_metrics(y_val, (fused > thr).astype(int), fused)
        log(f"  thr {thr:.2f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
            f"F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")
        rows.append({"model": f"rf_ae_or@{thr}", **m})
    log("  these are in-dataset numbers and do not select the operating point;")
    log("  the threshold must be re-selected by nested LOAO on this population")

    pd.DataFrame(rows).to_csv(METRICS, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {METRICS}")
    log(f"[saved] {REPORT}")
    log("=== UNIFIED TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
