# Dataset 7 Robustness Reproduction Notes for Future Codex

This file is written for a future Codex session. Read it before debugging or regenerating the Dataset 7 seminar-era robustness figures.

This folder is the GitHub-ready copy of the older `seminar_inspection` work. It intentionally contains only publishable/relevant material:

- cleaned notebooks;
- seminar-specific support utilities;
- selected figures;
- the original seminar report PDF;
- these reproduction notes.

Large training caches and full generated-figure dumps are intentionally not included.

## Current GitHub Folder

```text
Interpretable-Deep-Learning-Time-Series/
  dataset7_robustness_experiments/
    README.md
    REPRODUCTION_NOTES_FOR_CODEX.md
    notebooks/
      noise_sigma/
        noise_sigma_analysis.ipynb
      missing_values_imputation/
        missing_values_imputation_analysis.ipynb
      dynamics_change/
        dynamics_change_analysis.ipynb
    support_utils/
      src/
        utils.py
        utils_impute.py
        utils_DynamicsChange.py
    selected_figures/
      noise_sigma/
      missing_values_imputation/
      dynamics_change/
    seminar_report/
      semJosipDujmenovic.pdf
```

Expected clean workspace layout:

```text
workspace/
  DCIts/
  Interpretable-Deep-Learning-Time-Series/
```

The notebooks search upward for both this folder's `support_utils/` and the nearby original DCIts source containing `src/dcits.py`.

## Notebook Roles

### noise_sigma

Notebook:

```text
notebooks/noise_sigma/noise_sigma_analysis.ipynb
```

Support utility:

```text
support_utils/src/utils.py
```

Original selected source notebook:

```text
DCIts-DS7-L2-NSigma-testV002_HR_cache_py39.ipynb
```

Purpose: sweep Dataset 7 noise level and inspect prediction RMSE plus alpha stability/recovery.

Main parameters:

```python
seed = 1000
sigma_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
ts_length = 20000
n_runs = 5
window_length_gp = 5
temperature = 1.0
order = [1, 1]
FORCE_RECOMPUTE = False
```

Local regenerated cache/output path:

```text
notebooks/noise_sigma/artifacts/noise_sigma/data/
notebooks/noise_sigma/artifacts/noise_sigma/figures/
```

Important generated file:

```text
artifacts/noise_sigma/data/noise_sigma_summary.csv
```

Old local cache, if still available on Josip's machine:

```text
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\seminar_izlaz\rezultati\DCIts_DS7_L5_seed1000_ts20000_runs5.pkl
```

### missing_values_imputation

Notebook:

```text
notebooks/missing_values_imputation/missing_values_imputation_analysis.ipynb
```

Support utility:

```text
support_utils/src/utils_impute.py
```

Original selected source notebook:

```text
DCIts-DS7-L2-imputation-testV002_HR_cache_notebook_only_FULL_HR.ipynb
```

Purpose: test missing values in train/test sets and compare simple imputation methods.

Main parameters:

```python
seed = 1000
missing_ratios = [0.01, 0.02, 0.05, 0.10]
ts_length = 20000
n_runs = 5
window_length_gp = 5
temperature = 1.0
order = [1, 1]
```

Scenarios:

```text
missing_in_train
missing_in_test
```

Methods:

```text
linear
forward
mean
```

Local regenerated cache/output path:

```text
notebooks/missing_values_imputation/artifacts/missing_values_imputation/data/
notebooks/missing_values_imputation/artifacts/missing_values_imputation/figures/
```

Important generated files:

```text
artifacts/missing_values_imputation/data/missing_in_train_summary.csv
artifacts/missing_values_imputation/data/missing_in_test_summary.csv
artifacts/missing_values_imputation/data/combined_imputation_summary.csv
```

Old local cache folders, if still available:

```text
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\impute\Dataset 7\seed1000_w5\missing_in_train\ratio_...\<method>\data\stats.pkl
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\impute\Dataset 7\seed1000_w5\missing_in_test\ratio_...\<method>\data\stats.pkl
```

### dynamics_change

Notebook:

```text
notebooks/dynamics_change/dynamics_change_analysis.ipynb
```

Support utility:

```text
support_utils/src/utils_DynamicsChange.py
```

Original selected source notebook:

```text
DCIts-DS7-L2-DynamicsChange-testV2.ipynb
```

Purpose: generate a two-regime Dataset 7 dynamics-change case and inspect temporal/local alpha behavior.

Main parameters:

```python
seed = 1000
n_runs = 5
window_length_gp = 5
temperature = 1.0
order = [1, 1]
USE_CACHE = True
```

Local regenerated cache/output path:

```text
notebooks/dynamics_change/artifacts/dynamics_change/data/
notebooks/dynamics_change/artifacts/dynamics_change/figures/
```

Important generated files:

```text
artifacts/dynamics_change/data/results.pkl
artifacts/dynamics_change/data/stats.pkl
artifacts/dynamics_change/data/meta.json
```

Old local cache files, if still available:

```text
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\dynamics\data\stats.pkl
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\dynamics\data\results.pkl
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\dynamics\data\meta.json
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\dynamics\data\time_series_cd.pt
C:\Users\dujme\Desktop\Seminar_venv\DCIts\examples\artifacts\dynamics\data\ts_A.pt
```

Warning: old `results.pkl` is about 320 MB. Do not add it to git. The small `stats.pkl` is enough for global/regime summary heatmaps. Temporal local-alpha plots usually need per-window/run sequences from `results.pkl` or a fresh notebook run.

## Running Notes

For clean artifact paths, start Jupyter from the specific notebook folder, for example:

```powershell
cd Interpretable-Deep-Learning-Time-Series\dataset7_robustness_experiments\notebooks\noise_sigma
jupyter lab noise_sigma_analysis.ipynb
```

Equivalent folders:

```text
notebooks/missing_values_imputation
notebooks/dynamics_change
```

If a notebook is opened from a different Jupyter root, it should still find imports, but generated `artifacts/` may appear under the Jupyter working directory instead of beside the notebook.

## Lag-Axis Rule

DCIts stores learned alpha lags in reverse order relative to these hand-written ground-truth tensors.

Before comparing estimated alpha to ground truth, flip the lag axis:

```python
estimated_physical_lags = np.flip(estimated_model_alpha, axis=-1)
```

For a target-specific matrix with shape `(source, lag)`, use:

```python
estimated_target_physical_lags = np.flip(estimated_alpha[target], axis=1)
```

For temporal alpha curves, the cleaned notebooks use:

```python
def flip_lag(lag, window_length):
    return window_length - 1 - lag
```

Here `lag=0` is physical lag 1, `lag=1` is physical lag 2, etc.

## Selected Figures Included in Git

Use these selected figures for quick visual examples and report discussion.

Noise sigma:

```text
selected_figures/noise_sigma/ALL_alpha_vs_sigma.png
selected_figures/noise_sigma/RMSE_vs_sigma.png
selected_figures/noise_sigma/alpha_sigma_0.1_alpha_src1.png
selected_figures/noise_sigma/alpha_sigma_3.0_alpha_src1.png
selected_figures/noise_sigma/bias_sigma_comparison_canva.png
selected_figures/noise_sigma/alpha_rmse_by_source/X1_rmse_alpha_vs_sigma.png
selected_figures/noise_sigma/alpha_rmse_by_source/X2_rmse_alpha_vs_sigma.png
selected_figures/noise_sigma/alpha_rmse_by_source/X3_rmse_alpha_vs_sigma.png
selected_figures/noise_sigma/alpha_rmse_by_source/X4_rmse_alpha_vs_sigma.png
selected_figures/noise_sigma/alpha_rmse_by_source/X5_rmse_alpha_vs_sigma.png
```

Missing-values imputation:

```text
selected_figures/missing_values_imputation/rmse_comparison.png
selected_figures/missing_values_imputation/rmse_comparison2.png
selected_figures/missing_values_imputation/X_1_missing_ratios_alphas.png
```

Dynamics change:

```text
selected_figures/dynamics_change/vremenski_nizovi_promjena_dinamike.png
selected_figures/dynamics_change/promjena_alpha_par_i3_j3_lag4_vs_i3_j4_lag4.png
selected_figures/dynamics_change/temporalna_alpha_i3_j3_lag4.png
selected_figures/dynamics_change/temporalna_alpha_i3_j4_lag4.png
selected_figures/dynamics_change/regime_A_target_X3_alpha_heatmap_corrected.png
selected_figures/dynamics_change/regime_A_target_X3_alpha_heatmap_corrected.pdf
selected_figures/dynamics_change/regime_B_target_X3_alpha_heatmap_corrected.png
selected_figures/dynamics_change/regime_B_target_X3_alpha_heatmap_corrected.pdf
```

## Dynamics Heatmap Warning

Do not use old plots named like:

```text
alfe_prvi_rezim_X3.*
alfe_drugi_rezim_X3.*
```

They may exist in old generated folders, but they were visually misleading because the target/source/physical-lag labeling was not explicit enough.

Use the corrected selected figures:

```text
regime_A_target_X3_alpha_heatmap_corrected.*
regime_B_target_X3_alpha_heatmap_corrected.*
```

These show target X3, rows as source series, and columns as physical lags.

Numerical check from the old dynamics cache:

- Regime A target X3 matches ground truth well after lag flip.
- Regime B is genuinely weaker/mixed; it is not just a plotting flip bug.

## Final Cleanup State

Before copying this package into the GitHub repo, the three cleaned notebooks passed a smoke check:

- JSON parsed correctly;
- stored cell outputs were cleared;
- execution counts were cleared;
- code cells parsed as valid Python;
- setup/import cells worked from each notebook folder;
- tiny Dataset 7 generation/windowing smoke checks passed.

No new requirement beyond the root `requirements.txt` was identified.

## Future Codex Checklist

When Josip asks to reproduce or debug one of these seminar plots:

1. Read this file first.
2. Identify the experiment folder and cleaned notebook.
3. Check `selected_figures/` to see the current final figure name.
4. Check local regenerated `notebooks/<experiment>/artifacts/.../data` for cache files.
5. If local cache is missing, check the old `Seminar_venv/DCIts/examples/...` cache paths above.
6. For alpha comparisons, always confirm whether the lag axis has been flipped to physical lag order.
7. For dynamics X3 heatmaps, do not use old `alfe_prvi_rezim_X3` or `alfe_drugi_rezim_X3` plots.
8. Do not commit large `results.pkl`, `.pt`, `.pth`, `.npz`, or generated `artifacts/` folders unless Josip explicitly asks.
