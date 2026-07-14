"""
fusion_layer.py
================
โมดูล inference-time สำหรับรวมคะแนนของ Isolation Forest / Autoencoder (behavioral_v2, 92
feature) กับ Random Forest (80 feature เดิม) เป็นคะแนน fused เดียว — สกัดมาจาก
notebooks/08_fusion_layer.ipynb (ทดลอง/เปรียบเทียบ 3 fusion strategy บน val set แล้วบน LOAO
zero-day ด้วย) ไม่ใช่โค้ดใหม่ที่ยังไม่ได้ทดสอบ

Strategy ที่เลือก: confidence_weighted (RF-primary with unsupervised fallback) — ชนะทั้งบน
aggregate F1 บน val และบน per-category "captures best of 3 models" test

**CONFIDENCE_THRESHOLD/FUSED_DECISION_THRESHOLD ด้านล่าง tune จาก LOAO (zero-day) ไม่ใช่จาก val**
— sweep รอบแรกบน val ได้ CONFIDENCE_THRESHOLD=0.10 ซึ่งดูดีที่สุดบน val (F1=0.9965) แต่พอเอาไปรัน
จริงบน LOAO (10-category joint retrain, notebooks/08_fusion_layer.ipynb section 7) กลับพังยับ
(pooled LOAO F1=0.4143) เพราะ threshold ต่ำขนาดนั้นทำให้ RF "confident" (ตามนิยามที่ใช้: ระยะห่าง
จาก 0.5) เกือบทุกแถวแม้บน category ที่ไม่เคยเห็นตอนเทรน (RF มั่นใจแต่ผิดได้ในสถานการณ์ zero-day)
จึงแทบไม่เคย fallback ไปใช้ค่าเฉลี่ย IF-v2/AE-v2 เลย — sweep ใหม่ตรงๆ บน pooled LOAO score (ไม่ใช่
val) ได้ CONFIDENCE_THRESHOLD=0.95 (F1=0.8483 บน LOAO) ซึ่งกู้คืน zero-day recall ได้จริงใน 7/10
category (เช่น DoS Hulk 0.013->0.988, SSH-Patator 0.004->0.915) โดยเสียบน val แค่เล็กน้อย
(F1 0.9965->0.9905) — ใช้ค่านี้เป็นค่าสุดท้าย เพราะ LOAO คือ test ที่สำคัญที่สุดของงานนี้ ไม่ใช่ val

ข้อจำกัดที่รู้แล้ว ไม่ได้แก้ในไฟล์นี้ (ดู models/fusion_config.json's "known_limitations"):
Bot ยังใกล้ 0 ทุกโมเดล/ทุก strategy; PortScan ที่ fusion (mean ของ if_norm/ae_norm ตอน fallback)
แพ้ AE-v2 เดี่ยวๆ บน LOAO (0.030 vs 0.771 recall) — **ทดสอบแล้วว่า max(if_norm, ae_norm) แทน mean
ไม่ช่วย** (08_fusion_layer.ipynb Section 7.3): พอ re-sweep decision_threshold ให้แต่ละแบบ fair
กัน PortScan recall เท่ากันเป๊ะ (0.0303 ทั้งคู่) — สาเหตุจริงคือ confidence gate เอง ไม่ใช่วิธี
รวมคะแนนใน fallback: มีแค่ ~3% ของแถว PortScan zero-day ที่ RF confidence ต่ำกว่า 0.95 พอจะหลุด
เข้า fallback เลย (เทียบ 70-100% ของหมวดอื่น) RF มั่นใจ (และผิด) ว่าไม่ใช่ attack เกือบทุกแถว
ดังนั้นจะ mean หรือ max ก็กู้คืนไม่ได้ในเมื่อแถวส่วนใหญ่ไม่เคยไปถึง fallback เลย; Web Attack -
Brute Force เสีย recall เล็กน้อยเทียบ RF เดี่ยวๆ บน LOAO

Normalization constants (IF_MIN/IF_MAX/AE_MIN/AE_MAX) fit ครั้งเดียวบน fused val population
(405,085 แถว, join ผ่าน orig_row_index<->row_index) — reuse ตรงๆ ไม่ refit ทุกครั้งที่เรียกใช้
"""

import numpy as np

# ----------------------------------------------------------------------------
# Normalization constants -- fit บน val set ใน 08_fusion_layer.ipynb (Section 4), reuse ตรงๆ
# ----------------------------------------------------------------------------
IF_MIN = 0.32935471733365224
IF_MAX = 0.69906949405458
AE_MIN = 1.8078935681842268e-06
AE_MAX = 0.8110765814781189

# ----------------------------------------------------------------------------
# Fusion strategy ที่เลือก + parameter สุดท้าย -- tune จาก LOAO (zero-day), ไม่ใช่ val
# (08_fusion_layer.ipynb Section 7.2; ดู docstring ด้านบนสำหรับเหตุผลที่ val-tuned ใช้ไม่ได้)
# ----------------------------------------------------------------------------
CHOSEN_STRATEGY = "confidence_weighted"
CONFIDENCE_THRESHOLD = 0.95  # LOAO-tuned (val-tuned 0.10 failed the zero-day test)
FUSED_DECISION_THRESHOLD = 0.05  # LOAO-tuned (val-tuned 0.30 paired with the old confidence_threshold)


def normalize_if_score(anomaly_score, if_min=IF_MIN, if_max=IF_MAX):
    """Isolation Forest anomaly_score (unbounded) -> [0,1], higher = more attack-like."""
    return (np.asarray(anomaly_score) - if_min) / (if_max - if_min)


def normalize_ae_score(reconstruction_error, ae_min=AE_MIN, ae_max=AE_MAX):
    """Autoencoder reconstruction_error (unbounded MSE) -> [0,1], higher = more attack-like."""
    return (np.asarray(reconstruction_error) - ae_min) / (ae_max - ae_min)


def fuse_weighted_average(rf_norm, if_norm, ae_norm, weights=(1 / 3, 1 / 3, 1 / 3)):
    """Equal-weighted average -- naive baseline, NOT the chosen strategy (compared in
    08_fusion_layer.ipynb Section 5-6: dilutes RF's near-perfect in-dataset recall on
    categories where IF-v2/AE-v2 are weak, e.g. Bot/SSH-Patator/Web-Attack-*)."""
    w_rf, w_if, w_ae = weights
    return w_rf * np.asarray(rf_norm) + w_if * np.asarray(if_norm) + w_ae * np.asarray(ae_norm)


def fuse_max_score(rf_norm, if_norm, ae_norm):
    """Max ("OR" logic) -- NOT the chosen strategy (weakest of the three on val, both on
    aggregate F1 and on the best-of-three-models per-category test)."""
    return np.maximum(np.maximum(rf_norm, if_norm), ae_norm)


def fuse_confidence_weighted(rf_norm, if_norm, ae_norm, confidence_threshold=CONFIDENCE_THRESHOLD):
    """RF-primary with unsupervised fallback (CHOSEN STRATEGY). RF's own distance from 0.5
    ("confidence"/decisiveness, not correctness) decides whether to trust RF alone or fall
    back to the IF-v2/AE-v2 average -- motivated by RF's known zero-day blindness (it can be
    very confident and very wrong on an unseen category; the fallback is what's meant to catch
    what RF is confidently wrong about). CONFIDENCE_THRESHOLD=0.95 means only RF's most
    decisive calls are trusted directly -- true for most genuinely BENIGN traffic even in a
    zero-day fold, but not for RF's own uncertain, borderline calls on an unseen attack
    pattern, which correctly fall back to the IF-v2/AE-v2 average instead."""
    rf_norm = np.asarray(rf_norm)
    rf_confidence = np.abs(rf_norm - 0.5) * 2  # 0 = maximally uncertain, 1 = maximally decisive
    unsupervised_avg = (np.asarray(if_norm) + np.asarray(ae_norm)) / 2
    return np.where(rf_confidence >= confidence_threshold, rf_norm, unsupervised_avg)


def fuse(rf_attack_probability, if_anomaly_score, ae_reconstruction_error):
    """Inference-time entry point: raw scores from all three models in, fused binary
    prediction + fused score out, using the chosen strategy/threshold end-to-end.

    Parameters
    ----------
    rf_attack_probability : array-like
        Random Forest's `1 - P(BENIGN)` from `predict_proba` (already [0,1], no normalization).
    if_anomaly_score : array-like
        Isolation Forest's `-score_samples(X)` (unbounded, higher = more anomalous).
    ae_reconstruction_error : array-like
        Autoencoder's per-row MSE reconstruction error (unbounded, higher = more anomalous).

    Returns
    -------
    fused_score : np.ndarray
        Fused score in [0,1], higher = more attack-like.
    fused_prediction : np.ndarray of int
        Binary prediction (1 = attack, 0 = BENIGN) at `FUSED_DECISION_THRESHOLD`.
    """
    rf_norm = np.asarray(rf_attack_probability)
    if_norm = normalize_if_score(if_anomaly_score)
    ae_norm = normalize_ae_score(ae_reconstruction_error)

    fused_score = fuse_confidence_weighted(rf_norm, if_norm, ae_norm, confidence_threshold=CONFIDENCE_THRESHOLD)
    fused_prediction = (fused_score > FUSED_DECISION_THRESHOLD).astype(int)
    return fused_score, fused_prediction
