# MACLO-SiamNet v2 — Phase-1 Redesign

This branch is the auditable redesign aligned with the Phase-0 manuscript audit.

## Core mechanisms
1. Modality-specific stems plus a shared hierarchical encoder.
2. SCCT v2: compact Transformer fusion over modality/context tokens.
3. SPAM v2: task-specific sparse peer attention.
4. MACLO v2: gradient-level task coordination with pairwise cosine logging.
5. Explicit task masks for missing labels.
6. Explicit modality masks for unavailable images.

## Stroke/normal image + metadata dataset
The provided Excel/image package was audited locally before adding the public code:
- 649 metadata rows: 400 normal and 249 stroke.
- PNG images matched all 649 metadata records.
- JPG images matched 256 records.
- Two malformed Excel IDs are normalized: `1)97) -> 1(97)` and `2)204) -> 2(204)`.
- Two image IDs have no matching metadata record: `1(401)` and `2(5)`.
- File extension is **not** interpreted as MRI/CT provenance. PNG/JPG remain generic image sources until modality identity is independently verified.

Patient data and image files are intentionally not committed to this public repository.

### Build the local master index
```bash
python build_master_index.py \
  --excel "/path/Stroke and normal brain dataset.xlsx" \
  --images-zip "/path/dataset paper2.zip" \
  --extract-to "/path/extracted_images" \
  --output-dir generated_index
```

This produces `master_dataset.csv`, `linkage_audit.json`, and five patient-level fold files. Each fold uses 60% train, 20% validation, and 20% test by assigning three outer folds to training, one to validation, and one to testing.

### Classification experiment
The default classification config uses the PNG image plus structured metadata:
```bash
python train_v2.py \
  --csv generated_index/fold_0.csv \
  --config configs/classification_paper2.yaml
```

Numeric metadata normalization is fitted on the training split only. Missing BMI or other numeric values receive an explicit missingness indicator. Classification is binary Stroke vs Normal, matching the supplied `Brain Stroke` label.

## Important
This branch is a research implementation under controlled re-validation. Legacy manuscript numbers are not reproduced by this code until the new experiments are run and logged.


## Five-fold classification evaluation

Three controlled ablations are now supported:

```bash
python run_classification_cv.py --experiment image_only
python run_classification_cv.py --experiment metadata_only
python run_classification_cv.py --experiment image_metadata
```

Each experiment writes per-fold `metrics.json` and `predictions.csv`, then aggregates:
- Accuracy
- Sensitivity
- Specificity
- Precision
- F1
- ROC-AUC
- PR-AUC
- MCC
- Cohen's kappa

The aggregate file reports mean ± sample standard deviation across the five patient-level folds.

For image-based models, model selection is performed on validation AUC only. The test split is evaluated after checkpoint selection and is never used for early stopping.


## Metadata leakage audit

The supplied metadata were checked before using them as predictive inputs. The `d_dimer`
field is deterministically aligned with the class label in this dataset (400/400 normal cases
are recorded as Negative and 249/249 stroke cases as Positive). Because this behaves like a
label shortcut in the present dataset, `d_dimer` is excluded from the primary image+metadata
model. The primary clinical vector is therefore 29-dimensional.

Run the audit with:

```bash
python audit_metadata_leakage.py --csv generated_index/fold_0.csv
```

A D-dimer-inclusive result, if ever reported, must be clearly labeled as a sensitivity analysis
and not used as the principal evidence for multimodal classification performance.

## Sanity run before full CV

A short three-epoch config is provided only to verify data loading and optimization:

```bash
python train_v2.py \
  --csv generated_index/fold_0.csv \
  --config configs/classification_sanity.yaml \
  --checkpoint sanity_fold0.pt
```

Do not report the sanity-run metrics in the manuscript. Use the 256x256 full configs and all
five held-out test folds for the final results.
