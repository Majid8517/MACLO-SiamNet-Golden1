# MACLO-SiamNet-Golden

**competitive with SOTA in accuracy while superior in efficiency and multi-task capability.**

### Dataset Access

This repository does not include the CPAISD and AISD datasets due to clinical privacy constraints.

- CPAISD: Request access from the Regional Vascular Center, St. Petersburg.
  Contact: clinical authority / IRB / Data Use Agreement required.

- AISD: Available from the corresponding clinical institution upon request.

- ISLES 2018: Public:
  https://www.isles-challenge.org/ISLES2018/

Place acquired data into:
data/CPAISD/
data/AISD/
data/ISLES/
Our repository includes all scripts necessary to reproduce the experiments once the datasets are acquired.

## Requirements
- Python 3.12
- TensorFlow 2.15.0
- CUDA 11.8 + cuDNN 8.6 (for GPU acceleration)
- Minimum GPU memory: 8 GB
- Tested on: NVIDIA RTX 4050 (8 GB, Windows 10)

## Quick start
### Reproducing the results (single command)

conda env create -f environment.yml
conda activate maclo_golden

bash reproduce_all.sh
This script automatically:
- Downloads public ISLES data
- Configures folder structure
- Trains MACLO-SiamNet on ISLES / CPAISD / AISD
- Evaluates segmentation, classification, and lesion-age estimation
- Generates Tables 3–7 and Figures 3–9 in the paper
