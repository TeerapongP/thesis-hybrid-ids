"""
fusion_layer.py -- inference-time fusion for the ML ensemble layer.

STRATEGY: `rf_ae_or` -- a decision-level OR of Random Forest and Autoencoder-v2:

    fused_score = max(rf_attack_probability, ae_norm)
    attack      = fused_score > FUSED_DECISION_THRESHOLD

Isolation Forest is deliberately NOT an OR member. Its LOAO false-positive rate is
~0.095 (reports/fusion_layer_loao_trainnorm_results_*.csv), roughly 10x RF's and AE-v2's,
so ORing it in drags the whole system's FPR up for very little recall. It stays available
through `fuse_weighted_average`/`fuse_max_score`/`fuse_confidence_weighted` below, which
are kept only so the superseded strategies stay reproducible -- none of them is deployed.

WHY OR REPLACED THE `confidence_weighted` CASCADE (decided from nested LOAO selection on
train-fold-normalized scores; see reports/fusion_layer_loao_trainnorm_or_comparison.csv):

    candidate                      nested LOAO F1   recall     FPR
    OR: max(RF, AE)                       0.9253    0.9122   0.0079   <-- deployed
    OR: RF>t1 or AE>t2 (2 thresholds)     0.9126    0.8875   0.0076
    AE-v2 solo                            0.8896    0.8466   0.0075
    RF solo                               0.8327    0.7705   0.0106
    fusion: confidence_weighted           0.8324    0.7710   0.0108   <-- superseded
    IF-v2 solo                            0.4410    0.5481   0.1053

RF and AE-v2 are category specialists -- RF owns FTP-Patator / Web Attack - Brute Force /
DoS slowloris, AE-v2 owns PortScan / DDoS. A cascade can only pick ONE of them per row, and
its own tuner selected confidence_threshold=1.00 in 9/10 nested folds, i.e. it switched the
RF-primary design off. An OR fires when EITHER specialist is confident, which is why it
gains +3.6pp F1 over the best solo model at essentially unchanged FPR. The 2-threshold
variant has more freedom but generalizes WORSE across folds (0.9126), so the single-threshold
form is both better and simpler.

NORMALIZATION CONSTANTS ARE TRAIN-FITTED, AND THAT IS LOAD-BEARING. They are fit on
data/features/behavioral/cicids2017_behavioral_features_train.parquet (1,483,957 rows) with
the deployed models, matching what each LOAO fold does with its own train fold. The previous
constants were fit on the val fused population and had AE_MAX = 0.811 -- 6.6x the true
train-fit value -- which would shrink every deployed `ae_norm` by the same factor and make a
LOAO-tuned threshold unreachable, silencing the detector with no error. Do not substitute
val- or test-fit constants here.

FUSED_DECISION_THRESHOLD = 0.05 is LOAO-selected, not val-selected; nested selection picked
the same value in every fold, so it is stable rather than fold-specific.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Normalization constants -- fit on the behavioral TRAIN split with the deployed
# models (reports/deploy_norm_constants_trainfit.json). See docstring.
# ----------------------------------------------------------------------------
IF_MIN = 0.3288020239277581
IF_MAX = 0.6969734130072542
AE_MIN = 1.7595783674551058e-06
AE_MAX = 0.1224876269698143

# ----------------------------------------------------------------------------
# Deployed strategy + its LOAO-selected threshold
# ----------------------------------------------------------------------------
CHOSEN_STRATEGY = "rf_ae_or"
FUSED_DECISION_THRESHOLD = 0.05

# Superseded `confidence_weighted` parameter, kept so that strategy stays reproducible.
CONFIDENCE_THRESHOLD = 0.1


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
def fuse_rf_ae_or(rf_norm, ae_norm):
    """DEPLOYED STRATEGY. Element-wise max of RF's attack probability and AE-v2's
    normalized reconstruction error -- a decision-level OR once thresholded."""
    return np.maximum(np.asarray(rf_norm), np.asarray(ae_norm))


# ----------------------------------------------------------------------------
# Superseded strategies -- kept for reproducibility only, none is deployed
# ----------------------------------------------------------------------------
def fuse_weighted_average(rf_norm, if_norm, ae_norm, weights=(1 / 3, 1 / 3, 1 / 3)):
    """SUPERSEDED (nested LOAO F1 0.7457)."""
    w_rf, w_if, w_ae = weights
    return w_rf * np.asarray(rf_norm) + w_if * np.asarray(if_norm) + w_ae * np.asarray(ae_norm)


def fuse_max_score(rf_norm, if_norm, ae_norm):
    """SUPERSEDED (nested LOAO F1 0.5662). Three-way max INCLUDING Isolation Forest -- not
    the deployed two-way `fuse_rf_ae_or`; IF-v2's ~0.095 LOAO FPR is what sinks it."""
    return np.maximum(np.maximum(rf_norm, if_norm), ae_norm)


def fuse_confidence_weighted(rf_norm, if_norm, ae_norm, confidence_threshold=CONFIDENCE_THRESHOLD):
    """SUPERSEDED (nested LOAO F1 0.8324). The RF-primary cascade; see docstring for why it
    was replaced."""
    rf_norm = np.asarray(rf_norm)
    rf_confidence = np.abs(rf_norm - 0.5) * 2
    unsupervised_avg = (np.asarray(if_norm) + np.asarray(ae_norm)) / 2
    return np.where(rf_confidence >= confidence_threshold, rf_norm, unsupervised_avg)


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
        Isolation Forest's `-score_samples(X)`. Accepted and normalized for the returned
        metadata, but NOT used by the deployed OR rule.
    ae_reconstruction_error : array-like
        Autoencoder-v2's per-row MSE reconstruction error.

    Returns
    -------
    (fused_score, fused_prediction)
    """
    fused_score, fused_prediction, _, _ = fuse_with_metadata(
        rf_attack_probability, if_anomaly_score, ae_reconstruction_error
    )
    return fused_score, fused_prediction


def fuse_with_metadata(rf_attack_probability, if_anomaly_score, ae_reconstruction_error):
    """As `fuse`, plus which member actually triggered each row.

    Returns
    -------
    fused_score : np.ndarray
    fused_prediction : np.ndarray of int (1 = attack)
    margin : np.ndarray
        `fused_score - FUSED_DECISION_THRESHOLD`; distance from the decision boundary.
        This replaces the cascade's RF-decisiveness "confidence", which no longer exists
        under an OR rule -- do not read it as a calibrated probability.
    model_used : np.ndarray of str
        "random_forest" or "autoencoder_v2" -- whichever member supplied the max for that
        row (RF wins exact ties). Useful for the LLM explanation layer and for alert triage.
    """
    rf_norm = np.asarray(rf_attack_probability, dtype=float)
    if_norm = normalize_if_score(if_anomaly_score)  # metadata only, not in the OR
    ae_norm = normalize_ae_score(ae_reconstruction_error)

    fused_score = fuse_rf_ae_or(rf_norm, ae_norm)
    fused_prediction = (fused_score > FUSED_DECISION_THRESHOLD).astype(int)
    margin = fused_score - FUSED_DECISION_THRESHOLD
    model_used = np.where(rf_norm >= ae_norm, "random_forest", "autoencoder_v2")
    return fused_score, fused_prediction, margin, model_used
