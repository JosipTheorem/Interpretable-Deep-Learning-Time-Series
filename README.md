# Interpretable Deep Learning for Time Series with DCIts

This repository contains thesis experiment code for interpretable deep learning on time series using DCIts.

It includes:

- synthetic DCIts stability experiments;
- three reproducible experiment pipelines for the additional DCIts tasks;
- JSON configurations, CSV-producing scripts, and plotting code.

The experiments build on the original DCIts implementation:

https://github.com/hc-xai/dcits

## Repository Layout

```text
.
|-- synthetic_stability_experiments/
|   |-- hidden_driver_pipeline.py
|   |-- regime_change_pipeline.py
|   |-- smooth_coefficient_pipeline.py
|   |-- hidden_driver_config.json
|   |-- regime_change_config.json
|   |-- smooth_coefficient_config.json
|   |-- hidden_driver_analysis.ipynb
|   |-- regime_change_analysis.ipynb
|   |-- smooth_coefficient_analysis.ipynb
|   `-- komande.txt
|-- dcits_support/
|   `-- src/utils.py
|-- requirements.txt
|-- THIRD_PARTY.md
`-- LICENSE
```

## Setup

Clone the original DCIts repository:

```powershell
git clone https://github.com/hc-xai/dcits.git DCIts
```

Create and activate a Python environment, then install the required packages:

```powershell
pip install -r requirements.txt
```

Copy the thesis experiment folder into the DCIts examples folder:

```powershell
Copy-Item -Recurse -Force .\synthetic_stability_experiments .\DCIts\examples\synthetic_stability_experiments
```

Copy the support version of `utils.py` into the DCIts source folder:

```powershell
Copy-Item -Force .\dcits_support\src\utils.py .\DCIts\src\utils.py
```

The support file keeps the original DCIts utility interface, but also stores per-window `alpha`, `f`, and `C` sequences and MAE values needed by the thesis metrics.

## Running Experiments

The main commands are collected in:

```text
synthetic_stability_experiments/komande.txt
```

Typical pipeline runs:

```powershell
python DCIts\examples\synthetic_stability_experiments\hidden_driver_pipeline.py --config DCIts\examples\synthetic_stability_experiments\hidden_driver_config.json
python DCIts\examples\synthetic_stability_experiments\regime_change_pipeline.py --config DCIts\examples\synthetic_stability_experiments\regime_change_config.json
python DCIts\examples\synthetic_stability_experiments\smooth_coefficient_pipeline.py --config DCIts\examples\synthetic_stability_experiments\smooth_coefficient_config.json
```

The scripts save outputs as CSV tables and PDF figures. Large generated result folders are intentionally not tracked by git.

## Reproducibility

The experiment code records:

- fixed seeds;
- JSON configuration files;
- train/test metrics;
- interpretation metrics;
- false positive summaries;
- vector figures and CSV summaries.

For full reproduction, use the same Python environment and run the commands from `komande.txt`.

## License

This experiment code is released under the MIT License. See `LICENSE`.

The project depends on DCIts, which is also distributed under the MIT License. See `THIRD_PARTY.md`.
