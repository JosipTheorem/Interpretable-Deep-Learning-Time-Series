# Interpretable Deep Learning for Time Series with DCIts

This repository contains thesis experiment code for interpretable deep learning on time series using DCIts.

It includes:

- synthetic DCIts stability experiments;
- three reproducible experiment pipelines for the additional DCIts tasks;
- cleaned Dataset 7 robustness notebooks from the earlier seminar work;
- JSON configurations, CSV-producing scripts, and plotting code;
- small curated result/figure folders with representative outputs.

The experiments build on the original DCIts implementation:

https://github.com/hc-xai/dcits

## Repository Layout

```text
.
|-- selected_results/
|-- dataset7_robustness_experiments/
|   |-- notebooks/
|   |-- support_utils/
|   |-- selected_figures/
|   |-- seminar_report/
|   |-- README.md
|   `-- REPRODUCTION_NOTES_FOR_CODEX.md
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

## Final Pipeline Setup

This step is needed for the final pipeline-based experiments in `synthetic_stability_experiments/`. Copy that folder into the local DCIts `examples` folder:

```powershell
Copy-Item -Recurse -Force .\Interpretable-Deep-Learning-Time-Series\synthetic_stability_experiments .\DCIts\examples\synthetic_stability_experiments
```

Copy the support version of `utils.py` into the DCIts source folder:

```powershell
Copy-Item -Force .\Interpretable-Deep-Learning-Time-Series\dcits_support\src\utils.py .\DCIts\src\utils.py
```

The support file keeps the original DCIts utility interface, but also stores per-window `alpha`, `f`, and `C` sequences and MAE values needed by the thesis metrics.

The final pipeline notebooks should be opened and run from `DCIts/examples/synthetic_stability_experiments/`. The Dataset 7 robustness notebooks are different: they can be opened directly from `dataset7_robustness_experiments/` because they carry their own seminar support utilities.


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

## Selected Results

A lightweight subset of generated outputs is included in:

```text
selected_results/
```

This folder is small enough to keep in git and contains representative CSV tables and PDF figures from the three final pipeline groups. The complete local result folders are larger and are intentionally not tracked; they can be regenerated with the pipeline commands.

## Dataset 7 Robustness Notebooks

The earlier seminar-era Dataset 7 robustness work is included in:

```text
dataset7_robustness_experiments/
```

It contains cleaned notebooks for:

```text
noise_sigma
missing_values_imputation
dynamics_change
```

These notebooks can be run directly from this repository as long as the original `DCIts/` clone is next to it in the workspace. They use `dataset7_robustness_experiments/support_utils/` for the seminar-specific utility functions and write regenerated outputs to ignored local `artifacts/` folders.

Open the folder README for details:

```text
dataset7_robustness_experiments/README.md
```

A small curated image set is tracked in:

```text
dataset7_robustness_experiments/selected_figures/
```

## Running Final Pipeline Notebooks

After the setup and copy steps for `synthetic_stability_experiments/`, start Jupyter from the same environment.

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

The final pipeline notebooks expect to live inside `DCIts/examples/synthetic_stability_experiments/`, because they add `../..` to `sys.path` to import `src.utils`. Dataset 7 notebooks are documented separately in `dataset7_robustness_experiments/README.md`.

## Reproducibility

The experiment code records:

- fixed seeds;
- JSON configuration files;
- train/test metrics;
- interpretation metrics;
- false positive summaries;
- vector figures and CSV summaries.

For full reproduction of the final pipeline experiments, use the same Python environment and run the commands from `synthetic_stability_experiments/run_commands.txt`. For the Dataset 7 robustness notebooks, see `dataset7_robustness_experiments/README.md`.

## License

This experiment code is released under the MIT License. See `LICENSE`.

The project depends on DCIts, which is also distributed under the MIT License. See `THIRD_PARTY.md`.



