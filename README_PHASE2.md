# Phase 2 — Strong Architecture

Branch: `phase2-strong-architecture`

This phase intentionally does not overwrite the validated Phase-1 pipeline.

## Goal
Test whether a stronger, regularized image encoder and explicit image-clinical fusion improve performance and stability without relying on D-dimer or test-set tuning.

## Controlled ablations
1. `v3_concat.yaml` — simple image + clinical concatenation.
2. `v3_scct.yaml` — SCCT-v3 cross-context transformer.
3. `v3_scct_gate.yaml` — SCCT-v3 + adaptive reliability gating.

All three use:
- the same 29-D leakage-audited clinical input,
- the same image encoder family,
- the same 5 patient-level folds,
- validation-AUC checkpoint selection,
- label smoothing,
- stochastic depth,
- AdamW,
- cosine learning-rate decay.

## Fold-0 first
Run only Fold 0 before full cross-validation:

```bash
python train_v3_classifier.py \
  --csv generated_index/fold_0.csv \
  --config configs/v3_concat.yaml \
  --checkpoint results_v3/concat/fold_0/best.pt

python eval_v3_classifier.py \
  --csv generated_index/fold_0.csv \
  --config configs/v3_concat.yaml \
  --checkpoint results_v3/concat/fold_0/best.pt \
  --out-dir results_v3/concat/fold_0
```

Repeat with `v3_scct.yaml` and `v3_scct_gate.yaml`.

Do not select the final architecture using test results. Architecture decisions should use validation behavior and the predefined ablation plan; held-out test folds remain for final reporting.


## CCRF + Sparse Clinical-Conditioned Evidence Attention

The final classification-side architectural test adds sparse spatial evidence attention after
CCRF. Because the current stroke-vs-normal classification dataset has no lesion masks, this
module is deliberately not described as lesion-supervised attention.

```bash
python train_v3_classifier.py \
  --csv generated_index/fold_0.csv \
  --config configs/v3_ccrf_sparse.yaml \
  --checkpoint results_v3/ccrf_sparse/fold_0/best.pt

python eval_v3_classifier.py \
  --csv generated_index/fold_0.csv \
  --config configs/v3_ccrf_sparse.yaml \
  --checkpoint results_v3/ccrf_sparse/fold_0/best.pt \
  --out-dir results_v3/ccrf_sparse/fold_0
```

If the Fold-0 validation behavior is competitive with CCRF, run:

```bash
python run_v3_cv.py --experiment ccrf_sparse --skip-existing
```
