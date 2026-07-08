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
