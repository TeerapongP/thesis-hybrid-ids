# Digest fixes and machine learning code review — 2026-09-08

The two reported digest defects are fixed. The ML findings below remain open; resolving the evaluation findings requires new experiments, not editing existing result CSVs or substituting unvalidated thresholds.

## Completed changes

- `src/notifications/alert_dispatcher.py`: only a Gmail `sent` result acknowledges a digest. `error` and `dry_run` retain pending rows and return the corresponding status. Sender exceptions also leave rows pending.
- Only the primary keys actually included in the email are acknowledged. Alerts inserted during the send remain pending for the next digest. Updates run in one SQLite transaction with `executemany`, avoiding a large `IN (...)` parameter list.
- `tests/test_digest_regression.py`: offline tests use the actual dispatcher module, temporary SQLite databases, and a mocked sender. They cover error/dry-run retries, arrivals during delivery, raised exceptions, and an empty digest.
- `tests/test_alert_dispatcher.py`: updated the existing live smoke test's expectation to `dry_run` with pending rows. This script was inspected, not executed, because it sends real notifications and writes to the configured services.

Validation command: `python -m unittest discover -s tests -p test_digest_regression.py -v`.

Experiment ledger: the initial test harness left a SQLite inspection connection open on Windows; that test-only issue was corrected using `contextlib.closing`. The corrected harness failed on the original dispatcher in precisely the error/dry-run and concurrent-arrival cases, while the empty/exception controls passed. After the production change, all four tests passed. This agrees with the earlier reproduction: a failed send previously returned `flushed`, and a digest containing one alert acknowledged two when another arrived during sending.

Delivery remains at-least-once in failure scenarios: a process crash after Gmail accepts an email but before SQLite commits can cause a resend. Two simultaneous flush workers can also send the same selected batch. This patch fixes lost pending alerts; it does not introduce a cross-process delivery claim protocol.

## ML findings, in priority order

1. **[P1] The deployed fusion evaluation depends on the held-out test distribution.**

   `notebooks/08_fusion_layer.ipynb`, code cell 29 (zero-based), fits IF/AE score minima and maxima on each zero-day test fold. Cell 31 then selects thresholds on the pooled test labels. Inference in `src/modeling/fusion_layer.py:39` instead uses fixed validation-population normalization constants, with the legacy thresholds at lines 51–52. Consequently, the reported LOAO result is not an independent evaluation of the deployed normalization/threshold procedure.

   Recomputing confusion counts from all ten existing score parquet files at confidence 0.95 and decision 0.05 gave TP=318744, FP=7721, FN=106277, TN=3193599 and F1=0.8483032285, confirming the legacy figure. The later nested-selection CSV already documents selection optimism: its expanded-grid fusion F1 is 0.8552345 when tuned on all categories and 0.8150223 under category-held-out threshold selection. Those figures use a different grid from the legacy 0.8483 result. Both still inherit test-fold normalization; the notebook explicitly acknowledges this. The nested result is a diagnostic, not a fully independent replacement evaluation.

   `src/evaluation/fusion_loao_trainnorm.py` implements train-fold score normalization, but its checkpoint directory currently contains **zero score parquet files**. Complete that experiment, use an evaluation protocol that separates fitting/calibration/selection from the outer held-out category, and calibrate the final saved model pipeline consistently before replacing deployment constants. Preserve old results as historical experiments.

2. **[P1] Cross-dataset alignment does not align the feature transformations.**

   `src/features/feature_extractor.py:369` fits a separate scaler for each dataset. `src/evaluation/build_cross_dataset_eval.py:105` only intersects/reorders already-scaled columns. The same physical input therefore represents different values for training and external evaluation. From the saved processed training data, `Total Fwd Packets` ranges from 1 to 219759 for CICIDS-2017 and 1 to 20038 for UNSW. A 1000-packet flow becomes 0.00454591 in one representation and 0.04985776 in the other.

   For source-trained cross-dataset evaluation, fit shared-feature preprocessing on the source training data and apply that same transform to the target raw features. Column-name alignment alone is insufficient. Also exclude the four label-derived CTI columns: this script's `NON_FEATURE_COLS` at line 38 does not exclude them, although the actual model notebooks correctly do. No completed cross-dataset model score was validated in this review.

3. **[P1] The saved models lack their fitted preprocessing artifacts.**

   The feature extractors fit sentinel medians and scalers in memory, then save feature parquet files (`src/features/feature_extractor.py:424`, `src/features/feature_extractor_behavioral.py:407`). They do not serialize those fitted transforms. Model-saving cells in notebooks 02 and 06 save estimators/weights and AE hyperparameters, without bundling the ordered feature schema, fitted input scaler, and imputation state. No scaler/imputer artifact was found by filename search.

   This blocks reliable inference on fresh raw flows using the model artifacts alone. Fusion's IF/AE **score** normalization constants cannot replace the models' **input-feature** scalers. Persist versioned preprocessing state and exact feature order alongside each model, then check that processing a held-out raw batch reproduces its existing feature matrix and predictions.

4. **[P2] LOAO changes the training recipe as well as withholding an attack.**

   Notebook 08 cell 29 and `src/evaluation/fusion_loao_trainnorm.py:263` use 78 raw RF features and SMOTE before any RF scaling. The saved RF is trained on 80 features including two missingness flags, with SMOTE after network-feature scaling. Tree splits alone do not explain away this difference: changing feature scales before nearest-neighbor SMOTE changes the generated training samples.

   Additionally, notebook 06 trains the saved IF on the resampled feature file. Reading that file's `orig_row_index` confirmed 267786 synthetic rows among 1483957 total training rows. The joint LOAO IF instead fits the unresampled processed population. Thus the LOAO results evaluate related but different training recipes. Share preprocessing/feature selection/resampling code between final training and folds, fitting all state within each fold, and rerun before claiming the comparison isolates only unseen-category behavior.

5. **[P2] The fusion score is unbounded but the output promises a 0–100 risk score.**

   `src/modeling/fusion_layer.py:55` and its AE counterpart perform affine normalization without clipping, despite the `[0,1]` docstrings. With RF probability 0.5 and IF/AE raw values two fitted ranges above their minima, the actual `fuse()` returns score 2.0 and prediction 1. `src/notifications/alert_dispatcher.py:177` converts that into risk score 200. Below-range inputs can likewise yield negative scores.

   Keep the raw anomaly score available for ranking and thresholding, but define and test a bounded display-score mapping at the output boundary; correct the fusion docstrings. Do not silently clip the underlying experimental scores, because that changes ranking and potentially evaluation behavior. This is an additional finding, not part of the two requested digest fixes.

## Scope and limits

Reviewed the active feature-preparation scripts, original RF/IF/AE notebook feature selection, behavioral-v2 training/LOAO, fusion notebook including later comparison sections, inference module, train-normalized rerun, saved score files, and relevant model configs. Positive checks: the model notebooks exclude the label-derived CTI fields; main feature scaling is train-fitted; main val/test data are not SMOTE-resampled; the AE's model fitting uses BENIGN rows; fusion uses explicit row identifiers to join the different feature populations.

No model was retrained, no large training notebook was executed, and no existing model/checkpoint/metric file was replaced. The numerical checks used existing score files and narrow columns from processed datasets. The displayed nested-selection metrics were read from the saved report rather than rerunning that grid search. These checks establish the listed code/protocol defects; they do not predict corrected F1 or prove the full IDS pipeline is production-ready.
