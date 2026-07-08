# Interpretable Deep Learning for Time Series with DCIts

This repository is a research companion for experiments with [DCIts](https://github.com/hc-xai/dcits), an interpretable deep-learning method for multivariate time series. It contains experiment code, configurations, notebooks, and curated results; it is not a standalone replacement for the upstream DCIts project.

## Purpose and Outputs

The synthetic experiments test when DCIts local interpretation coefficients recover known dynamic relationships and when they reflect only predictive associations. The bead analysis applies the same approach to experimental particle trajectories; pipelines produce configuration records, CSV metrics, and PDF/PNG figures, with representative outputs tracked here.

## Start Here

Choose the part of the repository that matches your goal:

| Goal | Start here |
| --- | --- |
| Reproduce the final synthetic experiments | [Synthetic stability experiments](synthetic_stability_experiments/README.md) |
| Explore robustness on Dataset 7 | [Dataset 7 robustness experiments](dataset7_robustness_experiments/README.md) |
| Run the bead-tracking echo analysis | [Experimental bead analysis](experimental_bead_analysis/README.md) |
| Inspect representative synthetic outputs | [Selected results](selected_results/README.md) |

The synthetic pipelines, Dataset 7 notebooks, and bead analysis are separate workflows. You do not need to run all of them.

## Repository Guide

```text
synthetic_stability_experiments/  Three configurable synthetic experiment pipelines
dataset7_robustness_experiments/  Dataset 7 robustness notebooks and selected figures
experimental_bead_analysis/       Bead-tracking echo pipeline, notebook, sample data, and figures
selected_results/                 Curated outputs from the synthetic pipelines
dcits_support/                    DCIts utility support required by the synthetic pipelines
```

Each experiment folder has its own README and command list where applicable.

## Shared Setup

The code uses a local checkout of the upstream DCIts repository. Place the two repositories next to each other:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

From `workspace/`, clone the upstream project and this repository:

```powershell
git clone https://github.com/hc-xai/dcits.git DCIts
git clone https://github.com/JosipTheorem/Interpretable-Deep-Learning-Time-Series.git
```

Create and activate a Python environment, then install the upstream and repository-specific dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Or use Conda:

```powershell
conda create -n dcits-thesis python=3.10
conda activate dcits-thesis
```

With either environment active, install the dependencies:

```powershell
python -m pip install -r DCIts\requirements.txt
python -m pip install -r Interpretable-Deep-Learning-Time-Series\requirements.txt
```

Install Jupyter as well if you plan to run notebooks:

```powershell
python -m pip install jupyterlab notebook ipykernel
```

PyTorch is not pinned because its CPU/CUDA build depends on the machine. Install the appropriate build from the [official PyTorch selector](https://pytorch.org/get-started/locally/). The synthetic pipelines choose CUDA when available and otherwise use the CPU; pass `--device cpu` to force CPU execution.

## Synthetic Pipelines

Run the synthetic pipelines directly from this repository.

For example, from the repository root:

```powershell
python synthetic_stability_experiments\hidden_driver_pipeline.py --config synthetic_stability_experiments\hidden_driver_config.json
```

See [synthetic_stability_experiments/run_commands.txt](synthetic_stability_experiments/run_commands.txt) for smoke tests, full runs, and targeted sweeps.

## Results and Reproduction

The repository tracks representative tables and figures, not complete generated result trees or training caches. Regenerate full outputs with the commands documented in the relevant experiment folder.

Configurations, seeds, and summary tables are recorded by the pipelines. For a long-lived reproduction, also record the upstream DCIts commit hash and the Python/PyTorch versions used for the run.

## License and Attribution

This experiment code is released under the MIT License; see [LICENSE](LICENSE). DCIts is a separate MIT-licensed dependency. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and redistribution notes.
