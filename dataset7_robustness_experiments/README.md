# Dataset 7 Robustness Experiments

This folder contains the cleaned seminar-era Dataset 7 experiments used as supporting material for the thesis work on interpretable DCIts time-series models.

These notebooks are separate from the later pipeline-based experiments in `synthetic_stability_experiments/`. They study how DCIts interpretations behave under:

- increasing noise standard deviation;
- missing values and simple imputation methods;
- a piecewise change in the underlying dynamics.

## Folder Layout

```text
dataset7_robustness_experiments/
|-- notebooks/
|   |-- noise_sigma/
|   |   `-- noise_sigma_analysis.ipynb
|   |-- missing_values_imputation/
|   |   `-- missing_values_imputation_analysis.ipynb
|   `-- dynamics_change/
|       `-- dynamics_change_analysis.ipynb
|-- support_utils/
|   `-- src/
|       |-- utils.py
|       |-- utils_impute.py
|       `-- utils_DynamicsChange.py
|-- selected_figures/
|-- seminar_report/
|   `-- semJosipDujmenovic.pdf
|-- REPRODUCTION_NOTES_FOR_CODEX.md
`-- README.md
```

## Running the Notebooks

Use the same environment setup described in the repository root `README.md`. The original DCIts repository should be cloned next to this repository:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

The notebooks can be run directly from this repository. They do not need to be copied into `DCIts/examples/`, and no `dcits_support` copy step is needed for these seminar notebooks, because each notebook searches for both:

- `dataset7_robustness_experiments/support_utils/`; and
- the nearby original `DCIts/src/dcits.py` source file.

For the cleanest artifact paths, start Jupyter from the notebook folder you want to run. For example:

```powershell
cd Interpretable-Deep-Learning-Time-Series\dataset7_robustness_experiments\notebooks\noise_sigma
jupyter lab noise_sigma_analysis.ipynb
```

Equivalent folders exist for:

```text
notebooks/missing_values_imputation
notebooks/dynamics_change
```

Each notebook writes regenerated local outputs to an ignored `artifacts/` folder inside its own notebook directory.

## Selected Figures

The `selected_figures/` folder contains a small curated subset of figures suitable for GitHub and thesis/report discussion. Full generated result folders and large training caches are intentionally not included.

For dynamics-change heatmaps, use the corrected target-X3 figures:

```text
selected_figures/dynamics_change/regime_A_target_X3_alpha_heatmap_corrected.*
selected_figures/dynamics_change/regime_B_target_X3_alpha_heatmap_corrected.*
```

Older plots named like `alfe_prvi_rezim_X3` and `alfe_drugi_rezim_X3` were misleading because the target/source/physical-lag labeling was not explicit enough; they are not part of the selected figure set.

## Reproduction Notes

`REPRODUCTION_NOTES_FOR_CODEX.md` is a practical handoff file for future debugging/reproduction. It records:

- original source notebook names;
- support utility files;
- old local cache paths;
- selected figure names;
- lag-axis conventions;
- how to regenerate figures from cached `.pkl` files when available.

It is intentionally more operational than polished. Use it when a specific figure or result needs to be recreated later.

## Seminar Report

`seminar_report/semJosipDujmenovic.pdf` is the original seminar report connected to these experiments.
