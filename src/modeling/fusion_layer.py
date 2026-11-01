"""
fusion_layer.py -- inference-time fusion for the ML ensemble layer.

STRATEGY: `confidence_weighted` -- an RF-primary cascade with an unsupervised fallback:

    rf_confidence = |rf_attack_probability - 0.5| * 2
    fused_score   = rf_attack_probability          if rf_confidence >= CONFIDENCE_THRESHOLD
                    (if_norm + ae_norm) / 2        otherwise
    attack        = fused_score > FUSED_DECISION_THRESHOLD

Read as: trust Random Forest where it is decisive, and hand the undecided rows to the two
unsupervised detectors. All three members contribute, which is what makes this a hybrid
supervised/unsupervised decision rather than a supervised one with decoration.

SELECTED BY NESTED LOAO on the unified population (thresholds chosen on the other nine
categories and applied to the held-out one; reports/fusion_layer_loao_unified_or_comparison.csv):

    candidate                      nested LOAO F1   recall     FPR
    fusion: confidence_weighted           0.9782    0.9635   0.0007   <-- deployed
    OR: RF>t1 or AE>t2 (2 thresholds)     0.9745    0.9608   0.0011
    RF solo                               0.9737    0.9593   0.0011
    OR: max(RF, AE)                       0.9724    0.9559   0.0010   <-- superseded
    AND: RF and AE                        0.9504    0.9062   0.0001
    AE-v2 solo                            0.8736    0.8176   0.0055
    IF-v2 solo                            0.4371    0.4502   0.0619

This REVERSES the previous decision, and the reversal has a measured cause. `rf_ae_or` was
selected when Random Forest was confined to the 80-feature MachineLearningCSV matrix while
the unsupervised members had the 92-feature behavioral one; on that footing the OR beat RF
solo by 9.3pp. On the unified population all three members share the same 94 features, and
src/evaluation/rf_behavioral_ablation_loao.py isolated what that changed: removing just the
12 behavioral columns drops RF solo from F1 0.9737 to 0.8312, landing on the old joined
population's 0.8327. The behavioral columns, not the population, account for essentially the
whole gain. The old comparison was never fair to RF, and once RF has the same features the
OR's margin disappears entirely -- it now scores BELOW RF alone.

HOW MUCH THE ENSEMBLE ACTUALLY BUYS, stated plainly: 0.45pp F1 over RF solo (0.9782 vs
0.9737), with FPR 0.0007 against 0.0011. Better on both axes, so it is a real Pareto gain and
not a wash -- but it is a modest one, and any claim that the ensemble is decisively better
than its best single member is not supported on this population. Do not restate the old
9.3pp figure; it describes a configuration that no longer exists.

WHY THE CASCADE'S EARLIER DISMISSAL NO LONGER APPLIES. It was retired partly because its own
tuner chose confidence_threshold=1.00 in 9/10 nested folds, i.e. it switched the RF-primary
design off and degenerated. On the unified population the tuner chooses 0.95 in 9/10 folds
(0.1 in the PortScan fold), so the fallback path is genuinely carrying undecided rows. That
was checked before adopting it rather than assumed.

A side effect worth knowing: because the unified population gives IF and AE the SAME 94
features, the fallback's `(if_norm + ae_norm) / 2` averages two scores computed in one
feature space. Under the old two-population design it would have averaged across
incompatible spaces, which is part of what made the cascade hard to defend.

NORMALIZATION CONSTANTS ARE TRAIN-FITTED, AND THAT IS LOAD-BEARING. They are fit on
data/features/unified/cicids2017_unified_train.parquet (1,780,362 rows) with the deployed
models (reports/unified_norm_constants.json), matching what each LOAO fold does with its own
train fold, so a LOAO-selected threshold transfers. An earlier version of this file carried
val-fit constants with AE_MAX = 0.811 against a true train-fit 0.1225 -- 6.6x too large,
which would shrink every deployed `ae_norm` by that factor and make the threshold
unreachable, silencing the detector with no error. Never substitute val- or test-fit values.

KNOWN LIMITATIONS, per reports/fusion_layer_loao_unified_or_comparison.csv:
  * Bot nested recall 0.000 (RF solo 0.003, AE solo 0.017, IF solo 0.023). No member detects
    it, so no combination rule recovers it.
  * Isolation Forest is the single best detector on DoS Slowhttptest zero-day (0.839 vs
    AE's 0.241 and RF's 0.055 at argmax) but carries 0.09-0.15 FPR in every fold. The
    cascade uses it only on rows RF left undecided, which is the only way it earns a place.
  * Normalized scores are intentionally unclipped, so `fused_score` can exceed 1.0.
    Consumers must clamp at the presentation boundary.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Normalization constants -- fit on the unified TRAIN split with the deployed
# models (reports/unified_norm_constants.json). See docstring.
# ----------------------------------------------------------------------------
IF_MIN = 0.32810901309066537
IF_MAX = 0.71704325655017
AE_MIN = 5.890430657018442e-06
AE_MAX = 0.09661448001861572

# ----------------------------------------------------------------------------
# Deployed strategy + its LOAO-selected parameters
# ----------------------------------------------------------------------------
CHOSEN_STRATEGY = "confidence_weighted"
CONFIDENCE_THRESHOLD = 0.95
FUSED_DECISION_THRESHOLD = 0.03

# `model_used` values this module can emit. Callers that whitelist these strings -- notably
# alert_dispatcher.derive_model_used and shap_explain -- must be migrated together with any
# rename here, or every affected alert silently becomes "unknown".
MODEL_USED_VALUES = ("random_forest", "if_ae_fallback")


def normalize_if_score(anomaly_score, if_min=IF_MIN, if_max=IF_MAX):
    """Isolation Forest `-score_samples(X)` -> [0,1]-ish, higher = more attack-like.

    Not clipped: a zero-day row legitimately scores outside the train range, and clipping
    would destroy that signal. Callers that need a bounded 0-100 display score must clamp
    at the presentation boundary, not here.
    """
    return (np.asarray(anomaly_score) - if_min) / (if_max - if_min)


def normalize_ae_score(reconstruction_error, ae_min=AE_MIN, ae_max=AE_MAX):
    """Autoencoder per-row MSE -> [0,1]-ish, higher = more attack-like. Not clipped; see
    `normalize_if_score`."""
    return (np.asarray(reconstruction_error) - ae_min) / (ae_max - ae_min)


# ----------------------------------------------------------------------------
# Deployed fusion rule
# ----------------------------------------------------------------------------
def fuse_confidence_weighted(rf_norm, if_norm, ae_norm, confidence_threshold=CONFIDENCE_THRESHOLD):
    """DEPLOYED STRATEGY. RF-primary cascade with an unsupervised fallback.

    `rf_confidence` measures how far RF's probability sits from the undecided midpoint, so
    it is high at both ends (confidently BENIGN and confidently attack) and low near 0.5.
    Rows RF is decisive about keep RF's own score; the rest fall back to the mean of the two
    unsupervised scores.
    """
    rf_norm = np.asarray(rf_norm)
    rf_confidence = np.abs(rf_norm - 0.5) * 2
    unsupervised_avg = (np.asarray(if_norm) + np.asarray(ae_norm)) / 2
    return np.where(rf_confidence >= confidence_threshold, rf_norm, unsupervised_avg)


# ----------------------------------------------------------------------------
# Superseded strategies -- kept for reproducibility only, none is deployed
# ----------------------------------------------------------------------------
def fuse_rf_ae_or(rf_norm, ae_norm):
    """SUPERSEDED (nested LOAO F1 0.9724 on the unified population, below RF solo's 0.9737).
    Decision-level OR of RF and AE-v2. It was the deployed rule while RF was restricted to
    80 features; see the docstring for the ablation that explains why it lost its margin."""
    return np.maximum(np.asarray(rf_norm), np.asarray(ae_norm))


def fuse_weighted_average(rf_norm, if_norm, ae_norm, weights=(1 / 3, 1 / 3, 1 / 3)):
    """SUPERSEDED. Equal-weight mean of the three normalized scores."""
    w_rf, w_if, w_ae = weights
    return w_rf * np.asarray(rf_norm) + w_if * np.asarray(if_norm) + w_ae * np.asarray(ae_norm)


def fuse_max_score(rf_norm, if_norm, ae_norm):
    """SUPERSEDED. Three-way max INCLUDING Isolation Forest, whose 0.09-0.15 LOAO FPR is
    what sinks it as an independent trigger."""
    return np.maximum(np.maximum(rf_norm, if_norm), ae_norm)


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def fuse(rf_attack_probability, if_anomaly_score, ae_reconstruction_error):
    """Inference entry point.

    Parameters
    ----------
    rf_attack_probability : array-like
        Random Forest's `1 - P(BENIGN)` from `predict_proba` (already [0,1]).
    if_anomaly_score : array-like
        Isolation Forest's `-score_samples(X)`.
    ae_reconstruction_error : array-like
        Autoencoder's per-row MSE reconstruction error.

    Returns
    -------
    (fused_score, fused_prediction)
    """
    fused_score, fused_prediction, _, _ = fuse_with_metadata(
        rf_attack_probability, if_anomaly_score, ae_reconstruction_error
    )
    return fused_score, fused_prediction


def fuse_with_metadata(rf_attack_probability, if_anomaly_score, ae_reconstruction_error):
    """As `fuse`, plus which path decided each row.

    Returns
    -------
    fused_score : np.ndarray
    fused_prediction : np.ndarray of int (1 = attack)
    margin : np.ndarray
        `fused_score - FUSED_DECISION_THRESHOLD`; distance from the decision boundary.
        Not a calibrated probability -- do not present it as one.
    model_used : np.ndarray of str
        "random_forest" where RF was decisive enough to own the row, "if_ae_fallback" where
        the unsupervised average decided it. One of `MODEL_USED_VALUES`.
    """
    rf_norm = np.asarray(rf_attack_probability, dtype=float)
    if_norm = normalize_if_score(if_anomaly_score)
    ae_norm = normalize_ae_score(ae_reconstruction_error)

    fused_score = fuse_confidence_weighted(rf_norm, if_norm, ae_norm)
    fused_prediction = (fused_score > FUSED_DECISION_THRESHOLD).astype(int)
    margin = fused_score - FUSED_DECISION_THRESHOLD
    rf_decisive = np.abs(rf_norm - 0.5) * 2 >= CONFIDENCE_THRESHOLD
    model_used = np.where(rf_decisive, "random_forest", "if_ae_fallback")
    return fused_score, fused_prediction, margin, model_used
