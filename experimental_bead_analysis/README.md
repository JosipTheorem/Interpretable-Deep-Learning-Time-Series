# Experimental Bead Echo Analysis

This folder contains the experimental bead-tracking analysis that preceded the synthetic DCIts stability experiments. The work was inspired by Eva H.'s 2025 thesis and uses experimental particle-tracking echo data.

A tiny raw `.mat` sample is included so the code has a minimal runnable input. The full raw tracking dataset and Eva H.'s thesis PDF are not included because they are external materials and much larger.

## Contents

```text
experimental_bead_analysis/
|-- echo_analysis_pipeline.py
|-- run_commands.txt
|-- notebooks/
|   `-- multiple_run_one_echo_analysis.ipynb
|-- sample_data/
|   |-- lowrange echo 10 rad_s 1perc/
|   `-- highrange echo 10 rad_s 60perc/
|-- support_utils/
|   `-- src/
|       |-- util_echo.py
|       `-- utils_dipl.py
|-- selected_results/
`-- REPRODUCTION_NOTES_FOR_CODEX.md
```

## Main Artifacts

- `echo_analysis_pipeline.py` is the main reproducible script for the full echo-analysis workflow.
- `notebooks/multiple_run_one_echo_analysis.ipynb` is a cleaned, top-to-bottom sample-data notebook for loading the sample `.mat` files, regenerating lightweight diagnostics, running optional exploratory tests, preparing inline supervised `|delta raw_xy|` and `E(t)` experiments, and viewing selected results.
- `sample_data/` contains two 10-bead full-frame clusters: one from the 1% amplitude case and one from the 60% amplitude case.
- `support_utils/src/util_echo.py` contains bead/echo loading, clustering, plotting, statistics, and DCIts training helpers.
- `support_utils/src/utils_dipl.py` contains DCIts multi-run helpers used by the experimental bead analysis.
- `selected_results/` contains representative figures from `RezultatiV2` for low- and high-amplitude Echo 1 cases.

Older exploratory Echo 5 notebooks existed locally, but were not selected for GitHub because the later one-echo notebook and pipeline supersede them.

## Setup

Use the same sibling-repository layout as the rest of this project:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

Install the official DCIts dependencies first, then this repository's extra requirements:

```powershell
pip install -r DCIts\requirements.txt
pip install -r Interpretable-Deep-Learning-Time-Series\requirements.txt
```

The default pipeline data root is the included `sample_data/` folder. For full experiments, pass `--data-root` pointing to the complete local Eva tracking dataset.

## Running

From the repository root, a small non-training smoke run is:

```powershell
python experimental_bead_analysis\echo_analysis_pipeline.py --echo-limit 1 --cluster-limit 1 --skip-training --skip-acf-pacf
```

The full command list is in:

```text
experimental_bead_analysis/run_commands.txt
```

By default, generated outputs are written to:

```text
experimental_bead_analysis/artifacts/pipeline_results/
```

That folder is ignored by git.

## Selected Results

The selected results are intentionally compact and representative. Deep generated paths were flattened into short `figures/` folders where needed so Git on Windows can index them cleanly. They include:

- cluster overview plots;
- cluster time-series plots;
- DCIts alpha heatmaps for cluster 0;
- shuffled-control heatmaps;
- loss curves;
- self-alpha traces;
- ACF examples;
- event-conditioned interpretability plots;
- raw event relation line plots.

The high-amplitude selected-result folder also contains:

```text
selected_results/gamma_60perc_echo1/supervised_abs_delta_y/
```

Those figures come from the final notebook-only supervised mobility experiment, where DCIts inputs were trained to predict `|delta raw_xy|`.

A second notebook-extracted supervised event folder is also kept:

```text
selected_results/gamma_60perc_echo1/supervised_event_Et/
```

Those figures come from embedded outputs in the final `E(t)` cells of the original exploratory notebook. The local `drugi_opt_problem/E(t)` result folder was empty, so these plots were extracted directly from the notebook output cells.

The complete result folders in `Rezultati` and `RezultatiV2` are much larger and are not tracked.
