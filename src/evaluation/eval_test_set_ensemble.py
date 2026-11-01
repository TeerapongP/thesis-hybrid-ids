import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.evaluation.ensemble_eval_artifacts import (
    RUN_STARTED_AT_METADATA_KEY,
    RUN_STARTED_AT_REPORT_LABEL,
    build_ensemble_artifact_paths,
    ensure_artifact_paths_available,
    open_artifact_outputs_exclusively,
)
from src.modeling import fusion_layer as FL

ROOT = Path(__file__).resolve().parents[2]
reports_dir = ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
run_started_at, run_id, artifact_paths = build_ensemble_artifact_paths(
    reports_dir, FL.CONFIDENCE_THRESHOLD, FL.FUSED_DECISION_THRESHOLD
)
ensure_artifact_paths_available(artifact_paths)
metrics_path = artifact_paths["metrics"]
category_path = artifact_paths["per_category"]
report_path = artifact_paths["report"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run_relevant_changes_present():
    paths = [
        "src/evaluation/eval_test_set_ensemble.py",
        "src/modeling/fusion_layer.py",
        "models/fusion_config.json",
    ]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())

# One population, one feature list, one split. This script used to read two separate test
# matrices -- 80 features for RF, 92 for the unsupervised members -- and merge them on
# orig_row_index, a join that reconciled only 80.36% of the rows. build_unified_population.py
# removed the join, so there is nothing left to merge and no coverage caveat to report.
NON_FEATURE_COLS = ["Label","attack_cat","technique_id","technique_name","tactic",
                    "mapping_confidence","technique_id_encoded","tactic_encoded",
                    "mapping_confidence_encoded","is_known_technique","orig_row_index"]
numeric_feature_columns = lambda df, excluded: [
    c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

test_df = pd.read_parquet(ROOT / "data/features/unified/cicids2017_unified_test.parquet")
feature_cols = numeric_feature_columns(test_df, NON_FEATURE_COLS)
assert len(feature_cols)==94, len(feature_cols)
X = test_df[feature_cols].values
Xb = X.astype(np.float32)
rf = joblib.load(ROOT / "models/random_forest_unified.pkl")
t0=time.time(); proba = rf.predict_proba(X); rf_infer_s=time.time()-t0
benign_idx = list(rf.classes_).index("BENIGN")
rf_attack_prob = 1.0 - proba[:, benign_idx]
rf_pred_label = rf.classes_[proba.argmax(1)]
print(f"RF inference: {rf_infer_s:.1f}s for {len(test_df):,} rows -> {rf_infer_s/len(test_df)*1e6:.2f} us/flow")

iso = joblib.load(ROOT / "models/isolation_forest_unified.pkl")
t0=time.time(); if_score = -iso.score_samples(Xb); if_infer_s=time.time()-t0
print(f"IF-v2 inference: {if_infer_s:.1f}s -> {if_infer_s/len(Xb)*1e6:.2f} us/flow")

ae_config_path = ROOT / "models/autoencoder_unified_config.json"
cfg = json.loads(ae_config_path.read_text(encoding="utf-8"))
class AE(nn.Module):
    def __init__(s, d, h, b, act="relu"):
        super().__init__(); A={"relu":nn.ReLU}[act]
        ed=[d]+list(h)+[b]; dd=[b]+list(h)[::-1]+[d]
        el=[];
        for i in range(len(ed)-1):
            el.append(nn.Linear(ed[i],ed[i+1]))
            if i < len(ed)-2: el.append(A())
        s.encoder=nn.Sequential(*el)
        dl=[]
        for i in range(len(dd)-1):
            dl.append(nn.Linear(dd[i],dd[i+1]))
            if i < len(dd)-2: dl.append(A())
        s.decoder=nn.Sequential(*dl)
    def forward(s,x): return s.decoder(s.encoder(x))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = AE(cfg["input_dim"], cfg["hidden_dims"], cfg["bottleneck_dim"], cfg["activation"]).to(dev)
ae_weights_path = ROOT / "models/autoencoder_unified.pt"
ae.load_state_dict(torch.load(ae_weights_path, map_location=dev)); ae.eval()
errs=[]; t0=time.time()
with torch.no_grad():
    for i in range(0, len(Xb), 65536):
        xb = torch.tensor(Xb[i:i+65536]).to(dev)
        errs.append(torch.mean((ae(xb)-xb)**2, dim=1).cpu().numpy())
ae_err = np.concatenate(errs); ae_infer_s=time.time()-t0
print(f"AE-v2 inference ({dev}): {ae_infer_s:.1f}s -> {ae_infer_s/len(Xb)*1e6:.2f} us/flow")

scores = pd.DataFrame({"row_index": np.arange(len(test_df)), "Label": test_df["Label"].values,
                       "rf_attack_probability": rf_attack_prob,
                       "rf_pred_binary": (rf_pred_label!="BENIGN").astype(int),
                       "if_anomaly_score": if_score, "ae_reconstruction_error": ae_err})
print(f"\nTest population: {len(scores):,} rows, 100% of the test split (no join)")
y = (scores["Label"]!="BENIGN").astype(int).values

fused_score, fused_pred = FL.fuse(scores["rf_attack_probability"].values, scores["if_anomaly_score"].values, scores["ae_reconstruction_error"].values)
if_norm = FL.normalize_if_score(scores["if_anomaly_score"].values)
ae_norm = FL.normalize_ae_score(scores["ae_reconstruction_error"].values)

def metrics_row(name, yt, yp, sc):
    return dict(model=name, precision=precision_score(yt,yp,zero_division=0), recall=recall_score(yt,yp),
                f1=f1_score(yt,yp), auc_roc=roc_auc_score(yt,sc))

rows=[]
rows.append(metrics_row("Ensemble (rf_ae_or)", y, fused_pred, fused_score))
rows.append(metrics_row("RF solo (argmax, multi-class)", y, scores["rf_pred_binary"].values, scores["rf_attack_probability"].values))
AE_THR = cfg["chosen_threshold"]
rows.append(metrics_row("AE-v2 solo (99th pct thr)", y, (scores["ae_reconstruction_error"].values>AE_THR).astype(int), ae_norm))
if_pred = (iso.predict(Xb)==-1).astype(int)
rows.append(metrics_row("IF solo (contamination=0.2)", y, if_pred, if_norm))
res=pd.DataFrame(rows)
print("\n=== TEST SET (CICIDS-2017 unified), binary attack-vs-BENIGN, n={:,} ===".format(len(scores)))
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# There is no longer a "fused population vs. full population" split to report: every model
# scores every row, so RF solo above IS RF on the full test split.

from sklearn.metrics import f1_score as f1s
print("RF multi-class macro F1 (full test): {:.4f} | weighted F1: {:.4f}".format(
    f1s(test_df["Label"].values, rf_pred_label, average="macro"),
    f1s(test_df["Label"].values, rf_pred_label, average="weighted")))

scores["fused_pred"]=fused_pred
per = scores[scores["Label"]!="BENIGN"].groupby("Label").agg(n=("Label","size"), ensemble_recall=("fused_pred","mean"))
per["rf_recall"]=scores[scores["Label"]!="BENIGN"].groupby("Label")["rf_pred_binary"].mean()
print("\n=== per-category recall on TEST (ensemble vs RF solo) ===")
print(per.sort_values("n", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
ben = scores[scores["Label"]=="BENIGN"]
print(f"\nEnsemble FPR on test BENIGN: {ben['fused_pred'].mean():.6f}  (n={len(ben):,})")
print(f"RF solo   FPR on test BENIGN: {ben['rf_pred_binary'].mean():.6f}")
fusion_config_path = ROOT / "models/fusion_config.json"
fusion_config = json.loads(fusion_config_path.read_text(encoding="utf-8"))
assert fusion_config["chosen_strategy"] == FL.CHOSEN_STRATEGY
assert fusion_config["confidence_threshold"] == FL.CONFIDENCE_THRESHOLD
assert fusion_config["fused_decision_threshold"] == FL.FUSED_DECISION_THRESHOLD
for config_key, module_value in (
    ("if_min", FL.IF_MIN),
    ("if_max", FL.IF_MAX),
    ("ae_min", FL.AE_MIN),
    ("ae_max", FL.AE_MAX),
):
    assert fusion_config["normalization"][config_key] == module_value

provenance = {
    "run_id": run_id,
    RUN_STARTED_AT_METADATA_KEY: run_started_at.isoformat(),
    "dataset": "CICIDS-2017 test split",
    "fusion_strategy": FL.CHOSEN_STRATEGY,
    "confidence_threshold": FL.CONFIDENCE_THRESHOLD,
    "fused_decision_threshold": FL.FUSED_DECISION_THRESHOLD,
    "fusion_config_sha256": sha256_file(fusion_config_path),
    "autoencoder_config_sha256": sha256_file(ae_config_path),
    "evaluation_source_sha256": sha256_file(Path(__file__)),
    "fusion_source_sha256": sha256_file(Path(FL.__file__)),
    "git_head_revision": git_head_revision(),
    "run_relevant_worktree_changes_present": run_relevant_changes_present(),
    "test_data_sha256": sha256_file(ROOT / "data/features/unified/cicids2017_unified_test.parquet"),
    "random_forest_model_sha256": sha256_file(ROOT / "models/random_forest_unified.pkl"),
    "isolation_forest_model_sha256": sha256_file(ROOT / "models/isolation_forest_unified.pkl"),
    "autoencoder_weights_sha256": sha256_file(ae_weights_path),
    "ae_solo_threshold": AE_THR,
    "test_rows": len(test_df),
    "fused_test_rows": len(scores),
    "rf_feature_count": len(rf_cols),
    "feature_count": len(feature_cols),
}
for key, value in provenance.items():
    res[key] = value
    per[key] = value

report_sections = [
    "Ensemble test evaluation",
    f"Run ID: {run_id}",
    f"{RUN_STARTED_AT_REPORT_LABEL}: {run_started_at.isoformat()}",
    f"Dataset and split: {provenance['dataset']}",
    f"Fusion strategy: {FL.CHOSEN_STRATEGY}",
    f"Confidence threshold: {FL.CONFIDENCE_THRESHOLD}",
    f"Fused decision threshold: {FL.FUSED_DECISION_THRESHOLD}",
    f"Autoencoder solo threshold: {AE_THR}",
    f"Fusion config SHA-256: {provenance['fusion_config_sha256']}",
    f"Evaluation source SHA-256: {provenance['evaluation_source_sha256']}",
    f"Fusion source SHA-256: {provenance['fusion_source_sha256']}",
    f"Git HEAD revision: {provenance['git_head_revision']}",
    f"Run-relevant working tree changes present: {provenance['run_relevant_worktree_changes_present']}",
    f"Test data SHA-256: {provenance['test_data_sha256']}",
    f"Random Forest model SHA-256: {provenance['random_forest_model_sha256']}",
    f"Isolation Forest model SHA-256: {provenance['isolation_forest_model_sha256']}",
    f"Autoencoder weights SHA-256: {provenance['autoencoder_weights_sha256']}",
    f"Test rows (all models): {len(test_df)}",
    f"Fused joined test rows: {len(scores)}",
    f"RF feature count: {len(rf_cols)}",
    f"Feature count: {len(feature_cols)}",
    "",
    "Aggregate model metrics",
    res.drop(columns=list(provenance)).to_string(index=False, float_format=lambda value: f"{value:.6f}"),
    "",
    "Per-category recall on fused test population",
    per.drop(columns=list(provenance)).to_string(float_format=lambda value: f"{value:.6f}"),
    "",
    f"Ensemble false-positive rate on test BENIGN: {ben['fused_pred'].mean():.6f} (n={len(ben)})",
    f"RF false-positive rate on test BENIGN: {ben['rf_pred_binary'].mean():.6f}",
    f"Test rows: {len(scores)}",
]
with open_artifact_outputs_exclusively(artifact_paths) as outputs:
    res.to_csv(outputs["metrics"], index=False)
    per.to_csv(outputs["per_category"], index_label="Label")
    outputs["report"].write("\n".join(report_sections) + "\n")

print(f"Saved metrics: {metrics_path}")
print(f"Saved per-category results: {category_path}")
print(f"Saved run report: {report_path}")
