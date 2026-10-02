# MACLO-SiamNet v2 — Phase-1 Redesign

This branch is a clean, auditable redesign aligned with the Phase-0 manuscript audit.

## Scientific scope
The implementation no longer assumes paired MRI+CT for every patient. It is designed for heterogeneous stroke imaging with configurable modalities such as NCCT and CTP-derived maps (CBF, CBV, MTT, Tmax), plus optional structured metadata.

Core mechanisms:
1. **Modality-specific stems + shared encoder** for heterogeneous imaging.
2. **SCCT v2**: an actual compact Transformer over modality/context tokens.
3. **SPAM v2**: task-specific sparse peer attention over pooled spatial tokens.
4. **MACLO v2**: gradient-level multi-task coordination with pairwise cosine logging.
5. **Explicit task masks**: missing labels are never silently converted into valid targets.
6. **Missing-modality masks**: unavailable imaging channels are masked rather than fabricated.

## Current task policy
- Segmentation: enabled.
- Lesion-age regression: enabled when a valid target exists.
- Classification: optional and disabled by default until defensible class labels are fixed from source data.

## Important
This branch is a **research implementation under controlled re-validation**. Legacy manuscript numbers must not be treated as reproduced by this code. All final tables should be regenerated from fixed patient-level folds and saved logs.

## Quick smoke test
```bash
python smoke_test_v2.py
```

## Training
Prepare a CSV with patient-level rows and explicit modality paths. See `maclo_v2/dataset.py` for the schema.

```bash
python train_v2.py --csv data/splits_v2.csv --config configs/phase1.yaml
```

## Recommended dataset roles
- CPAISD: primary NCCT / clinical metadata / lesion-age target where valid.
- AISD: external NCCT segmentation validation.
- ISLES 2018: CTP-domain segmentation/generalization.

The exact final role of every field must be verified against the original dataset documentation before reporting results.
