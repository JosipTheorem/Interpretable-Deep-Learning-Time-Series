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
|   `-- run_commands.txt
|-- dcits_support/
|   `-- src/utils.py
|-- requirements.txt
|-- THIRD_PARTY.md
`-- LICENSE
```

## Setup

This repository does not vendor the full DCIts source code. Use it together with a local clone of the official DCIts repository.

Recommended folder layout:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

From an empty workspace folder, clone both repositories:

```powershell
git clone https://github.com/hc-xai/dcits.git DCIts
git clone https://github.com/JosipTheorem/Interpretable-Deep-Learning-Time-Series.git
```

Create and activate a Python environment. You can use either `venv` or `conda`.

Example with `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Example with `conda`:

```powershell
conda create -n dcits-thesis python=3.10
conda activate dcits-thesis
```

Install DCIts dependencies using the official DCIts instructions or its `requirements.txt`, then install the extra packages used by these thesis scripts:

```powershell
pip install -r DCIts\requirements.txt
pip install -r Interpretable-Deep-Learning-Time-Series\requirements.txt
```

For notebook use, also install Jupyter tools:

```powershell
pip install jupyterlab notebook ipykernel
```

## IMPORTANT: Copy this repository's experiment folder into the DCIts examples folder:

```powershell
Copy-Item -Recurse -Force .\Interpretable-Deep-Learning-Time-Series\synthetic_stability_experiments .\DCIts\examples\synthetic_stability_experiments
```

Copy the support version of `utils.py` into the DCIts source folder:

```powershell
Copy-Item -Force .\Interpretable-Deep-Learning-Time-Series\dcits_support\src\utils.py .\DCIts\src\utils.py
```

The support file keeps the original DCIts utility interface, but also stores per-window `alpha`, `f`, and `C` sequences and MAE values needed by the thesis metrics.

The notebooks should be opened and run from `DCIts/examples/synthetic_stability_experiments/`.


### CUDA

The pipelines use `--device auto` by default. If PyTorch sees a CUDA-capable GPU, DCIts will use CUDA; otherwise it falls back to CPU. You can also force CPU explicitly with `--device cpu`.

CUDA is not pinned in `requirements.txt`, because the correct PyTorch build depends on the operating system, driver, GPU, and CUDA version. For GPU runs, install PyTorch using the official selector:

https://pytorch.org/get-started/locally/

Then check the environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Running Experiments

The main commands are collected in:

```text
synthetic_stability_experiments/run_commands.txt
```

Typical pipeline runs:

```powershell
python DCIts\examples\synthetic_stability_experiments\hidden_driver_pipeline.py --config DCIts\examples\synthetic_stability_experiments\hidden_driver_config.json
python DCIts\examples\synthetic_stability_experiments\regime_change_pipeline.py --config DCIts\examples\synthetic_stability_experiments\regime_change_config.json
python DCIts\examples\synthetic_stability_experiments\smooth_coefficient_pipeline.py --config DCIts\examples\synthetic_stability_experiments\smooth_coefficient_config.json
```

The scripts save outputs as CSV tables and PDF figures. Large generated result folders are intentionally not tracked by git.

Add `--no-training-results` to any pipeline command if you want to keep the CSV tables and PDF figures but skip the large `training_results.pkl` bundles.

## Running Notebooks

After the setup and copy steps, start Jupyter from the same environment.

With classic Jupyter Notebook:

```powershell
jupyter notebook DCIts\examples\synthetic_stability_experiments
```

With JupyterLab:

```powershell
jupyter lab DCIts\examples\synthetic_stability_experiments
```

Open one of:

```text
hidden_driver_analysis.ipynb
regime_change_analysis.ipynb
smooth_coefficient_analysis.ipynb
```

The notebooks expect to live inside `DCIts/examples/synthetic_stability_experiments/`, because they add `../..` to `sys.path` to import `src.utils`.

## Reproducibility

The experiment code records:

- fixed seeds;
- JSON configuration files;
- train/test metrics;
- interpretation metrics;
- false positive summaries;
- vector figures and CSV summaries.

For full reproduction, use the same Python environment and run the commands from `run_commands.txt`.

## License

This experiment code is released under the MIT License. See `LICENSE`.

The project depends on DCIts, which is also distributed under the MIT License. See `THIRD_PARTY.md`.
