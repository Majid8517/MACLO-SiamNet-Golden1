# MACLO-SiamNet-Golden

**competitive with SOTA in accuracy while superior in efficiency and multi-task capability.**

Reproducible code for AIS segmentation + classification + lesion-age estimation.
The datasets used in this study (CPAISD, AISD, ISLES2018) are not publicly redistributable due to patient privacy regulations. Users must obtain access from their respective official sources. Our repository includes all scripts necessary to reproduce the experiments once the datasets are acquired.

## Quick start
```bash
conda env create -f environment.yml
conda activate maclo_golden

bash reproduce_all.sh
