# Experimental Bead-Tracking Echo Analysis

This folder contains a DCIts analysis workflow for experimental particle-tracking echo data. It provides a command-line pipeline, a sample-data notebook, a permitted compact experimental subset, and representative figures.

The complete raw data set is external and is not included in this repository. The included `sample_data/` folder is a small 20-bead subset approved for use in this GitHub repository so the workflow can be run without private local data.

## Contents

- `echo_analysis_pipeline.py` is the main command-line workflow.
- `notebooks/multiple_run_one_echo_analysis.ipynb` demonstrates the workflow on the included sample data.
- `sample_data/` contains two permitted 10-bead, full-frame clusters: one 1% amplitude case and one 60% amplitude case.
- `support_utils/src/` contains the data-loading, clustering, plotting, statistics, and DCIts training helpers.
- `selected_results/` contains representative low- and high-amplitude Echo 1 figures.

## Setup

Use the same sibling-repository layout as the rest of this project:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

Install the official DCIts dependencies first, then this repository's extra requirements:

```powershell
python -m pip install -r DCIts\requirements.txt
python -m pip install -r Interpretable-Deep-Learning-Time-Series\requirements.txt
```

The default pipeline data root is the included `sample_data/` folder. For full experiments, pass `--data-root` pointing to a local copy of the complete tracking data set.

## Run

From the repository root, a small non-training smoke run is:

```powershell
python experimental_bead_analysis\echo_analysis_pipeline.py --echo-limit 1 --cluster-limit 1 --skip-training --skip-acf-pacf
```

The full command list is in [run_commands.txt](run_commands.txt).

By default, generated outputs are written to:

```text
experimental_bead_analysis/artifacts/pipeline_results/
```

## Selected Results

The selected results include:

- cluster overview and time-series plots;
- DCIts alpha heatmaps and shuffled controls;
- loss curves and self-alpha traces;
- ACF examples;
- event-conditioned interpretability plots;
- raw event-relation plots.

The high-amplitude selected-result folder also contains two supervised analyses:

- `supervised_abs_delta_y/`, where DCIts inputs predict `|delta raw_xy|`;
- `supervised_event_Et/`, where DCIts inputs predict a binary event indicator.
