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

## Quick start
```bash
conda env create -f environment.yml
conda activate maclo_golden

bash reproduce_all.sh
