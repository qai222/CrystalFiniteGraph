Silicate Zeolite Dataset
---

## overview
- 137992 CSSR files provided by [Mohamad Moosavi](https://chem-eng.utoronto.ca/faculty-staff/faculty-members/seyed-mohamad-moosavi/).
- Target values were calculated using [zpp](https://www.zeoplusplus.org/).
- Features were calculated using [networkx](https://networkx.org/).

## prepare
Work folder: [prepare](prepare)
1. Download `cssr.tar.gz` and unzip it to `cssr` folder.
2. Run `zpp` on CSSR files using [batch_calculation.sh](prepare/batch_calculation.sh) or [batch_calculation_parallel.sh](prepare/batch_calculation_parallel.sh).
    This creates three subfolders `res`, `chan`, and `sa`. Precomputed results can be unzipped from [zpp_results.7z](prepare/zpp_results.7z).
3. Use [prepare.py](prepare/prepare.py) to export results.
