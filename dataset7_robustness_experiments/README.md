# Dataset 7 Robustness Experiments

This folder contains notebooks that study how DCIts interpretations behave under:

- increasing noise standard deviation;
- missing values and simple imputation methods;
- a piecewise change in the underlying dynamics.

## Running the Notebooks

Use the shared environment setup in the [repository README](../README.md). The original DCIts repository should be cloned next to this repository:

```text
workspace/
|-- DCIts/
`-- Interpretable-Deep-Learning-Time-Series/
```

The notebooks can run directly from this repository. They do not need to be copied into `DCIts/examples/`, and no `dcits_support` copy step is needed because each notebook searches for both:

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

The `selected_figures/` folder contains a curated subset of figures. Full generated result folders and large training caches are intentionally not included.

For dynamics-change heatmaps, use the corrected target-X3 figures:

```text
selected_figures/dynamics_change/regime_A_target_X3_alpha_heatmap_corrected.*
selected_figures/dynamics_change/regime_B_target_X3_alpha_heatmap_corrected.*
```
