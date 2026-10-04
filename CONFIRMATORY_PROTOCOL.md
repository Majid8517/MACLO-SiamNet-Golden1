# Frozen Confirmatory Evaluation Protocol

## Purpose
This stage is confirmatory rather than exploratory. The architecture has been frozen after the development-stage ablations.

## Frozen candidate and baseline
- Candidate: **CCRF + linear clinical encoder**
- Baseline: **simple concatenation + linear clinical encoder**
- Clinical feature set: leakage-audited 29-D vector with **D-dimer excluded**
- Decision threshold: **0.5**, fixed
- Model selection: validation AUC only
- No test-fold tuning, threshold optimization, feature editing, or architectural modification is permitted after confirmatory results are inspected.

## Data
- 649 patients
- Binary Stroke vs Normal classification
- Patient-level splitting
- The same cohort is used; therefore this is **internal confirmatory validation**, not external validation.

## Evaluation design
Repeated stratified 5-fold cross-validation with five pre-specified seeds:

1. 2026
2. 31415
3. 27182
4. 16180
5. 42424

This yields 25 test folds per model. Within each repeat, every patient appears in the test set exactly once.

The validation fold is deterministically defined as the next fold after the test fold.

## Primary metrics
- ROC-AUC
- F1
- MCC

## Secondary metrics
- Accuracy
- Sensitivity
- Specificity
- Precision
- PR-AUC
- Cohen's kappa

## Statistical comparison
After all repeats:
1. Average each patient's five OOF probabilities for each model.
2. Compute patient-level ROC-AUC, F1, and MCC.
3. Use paired patient-level bootstrap (10,000 resamples) for differences and 95% CIs.
4. Use exact McNemar test on thresholded predictions.
5. Do not claim external generalization from this analysis.

## Interpretation rule
The development-stage results are reported as ablations. The confirmatory repeated-CV results are the primary internal-validation results for the frozen candidate.
