import os, time, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.modeling import fusion_layer as FL

NFC_RF = ["Label","attack_cat","technique_id","technique_name","tactic","mapping_confidence",
          "technique_id_encoded","tactic_encoded","mapping_confidence_encoded","is_known_technique"]
NFC_B  = NFC_RF + ["orig_row_index"]
numeric_feature_columns = lambda df, excluded: [
    c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

# ---------- RF on original 80-feature test ----------
rf_test = pd.read_parquet("data/features/cicids2017_features_test.parquet")
rf_cols = numeric_feature_columns(rf_test, NFC_RF); assert len(rf_cols)==80, len(rf_cols)
rf = joblib.load("models/random_forest.pkl")
t0=time.time(); proba = rf.predict_proba(rf_test[rf_cols].values); rf_infer_s=time.time()-t0
benign_idx = list(rf.classes_).index("BENIGN")
rf_attack_prob = 1.0 - proba[:, benign_idx]
rf_pred_label = rf.classes_[proba.argmax(1)]
rf_df = pd.DataFrame({"row_index": np.arange(len(rf_test)), "Label": rf_test["Label"].values,
                      "rf_attack_probability": rf_attack_prob,
                      "rf_pred_binary": (rf_pred_label!="BENIGN").astype(int)})
print(f"RF inference: {rf_infer_s:.1f}s for {len(rf_test):,} rows -> {rf_infer_s/len(rf_test)*1e6:.2f} us/flow")

# ---------- IF-v2 + AE-v2 on 92-feature behavioral test ----------
b_test = pd.read_parquet("data/features/behavioral/cicids2017_behavioral_features_test.parquet")
b_cols = numeric_feature_columns(b_test, NFC_B); assert len(b_cols)==92, len(b_cols)
Xb = b_test[b_cols].values.astype(np.float32)

iso = joblib.load("models/isolation_forest_behavioral_v2.pkl")
t0=time.time(); if_score = -iso.score_samples(Xb); if_infer_s=time.time()-t0
print(f"IF-v2 inference: {if_infer_s:.1f}s -> {if_infer_s/len(Xb)*1e6:.2f} us/flow")

cfg = json.load(open("models/autoencoder_behavioral_v2_config.json"))
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
ae.load_state_dict(torch.load("models/autoencoder_behavioral_v2.pt", map_location=dev)); ae.eval()
errs=[]; t0=time.time()
with torch.no_grad():
    for i in range(0, len(Xb), 65536):
        xb = torch.tensor(Xb[i:i+65536]).to(dev)
        errs.append(torch.mean((ae(xb)-xb)**2, dim=1).cpu().numpy())
ae_err = np.concatenate(errs); ae_infer_s=time.time()-t0
print(f"AE-v2 inference ({dev}): {ae_infer_s:.1f}s -> {ae_infer_s/len(Xb)*1e6:.2f} us/flow")

b_df = pd.DataFrame({"orig_row_index": b_test["orig_row_index"].values,
                     "if_anomaly_score": if_score, "ae_reconstruction_error": ae_err})

# ---------- join ----------
joined_scores = b_df.merge(rf_df, left_on="orig_row_index", right_on="row_index", validate="1:1")
print(f"\nFused test population: {len(joined_scores):,} / RF full {len(rf_df):,} = {len(joined_scores)/len(rf_df)*100:.2f}%")
y = (joined_scores["Label"]!="BENIGN").astype(int).values

fused_score, fused_pred = FL.fuse(joined_scores["rf_attack_probability"].values, joined_scores["if_anomaly_score"].values, joined_scores["ae_reconstruction_error"].values)
if_norm = FL.normalize_if_score(joined_scores["if_anomaly_score"].values)
ae_norm = FL.normalize_ae_score(joined_scores["ae_reconstruction_error"].values)

def metrics_row(name, yt, yp, sc):
    return dict(model=name, precision=precision_score(yt,yp,zero_division=0), recall=recall_score(yt,yp),
                f1=f1_score(yt,yp), auc_roc=roc_auc_score(yt,sc))

rows=[]
rows.append(metrics_row("Ensemble (confidence_weighted)", y, fused_pred, fused_score))
rows.append(metrics_row("RF solo (argmax, multi-class)", y, joined_scores["rf_pred_binary"].values, joined_scores["rf_attack_probability"].values))
AE_THR = cfg["chosen_threshold"]
rows.append(metrics_row("AE-v2 solo (99th pct thr)", y, (joined_scores["ae_reconstruction_error"].values>AE_THR).astype(int), ae_norm))
if_pred = iso.predict(Xb); if_pred = (if_pred==-1).astype(int)
b_df["if_pred"]=if_pred
joined_if_predictions = b_df[["orig_row_index","if_pred"]].merge(rf_df[["row_index"]], left_on="orig_row_index", right_on="row_index")
rows.append(metrics_row("IF-v2 solo (contamination=0.2)", y, joined_if_predictions["if_pred"].values, if_norm))
res=pd.DataFrame(rows)
print("\n=== TEST SET (CICIDS-2017), binary attack-vs-BENIGN, fused population n={:,} ===".format(len(joined_scores)))
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# RF on its FULL test population (not just fused subset), for reference
yf = (rf_df["Label"]!="BENIGN").astype(int).values
print("\nRF solo on FULL test population (n={:,}): P={:.4f} R={:.4f} F1={:.4f} AUC={:.4f}".format(
    len(rf_df), precision_score(yf, rf_df["rf_pred_binary"]), recall_score(yf, rf_df["rf_pred_binary"]),
    f1_score(yf, rf_df["rf_pred_binary"]), roc_auc_score(yf, rf_df["rf_attack_probability"])))

# macro/weighted multi-class for RF
from sklearn.metrics import f1_score as f1s
print("RF multi-class macro F1 (full test): {:.4f} | weighted F1: {:.4f}".format(
    f1s(rf_test["Label"].values, rf_pred_label, average="macro"),
    f1s(rf_test["Label"].values, rf_pred_label, average="weighted")))

# per-category recall of ensemble
joined_scores["fused_pred"]=fused_pred
per = joined_scores[joined_scores["Label"]!="BENIGN"].groupby("Label").agg(n=("Label","size"), ensemble_recall=("fused_pred","mean"))
per["rf_recall"]=joined_scores[joined_scores["Label"]!="BENIGN"].groupby("Label")["rf_pred_binary"].mean()
print("\n=== per-category recall on TEST (ensemble vs RF solo) ===")
print(per.sort_values("n", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
# FPR
ben = joined_scores[joined_scores["Label"]=="BENIGN"]
print(f"\nEnsemble FPR on test BENIGN: {ben['fused_pred'].mean():.6f}  (n={len(ben):,})")
print(f"RF solo   FPR on test BENIGN: {ben['rf_pred_binary'].mean():.6f}")
res.to_csv(os.path.join("reports","ensemble_test_metrics.csv"), index=False)
per.to_csv(os.path.join("reports","ensemble_test_per_category.csv"))
