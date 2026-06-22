# Selected Results

This folder contains a compact subset of the full synthetic-pipeline output tree. It is intended for quick inspection without storing large generated data or training bundles.

Included here:

- summary CSV tables for all three synthetic stability tasks;
- representative hidden-driver comparison figures;
- regime-change examples for high/low SNR and source, lag, and sign changes;
- smooth-coefficient examples showing SNR contrast, sinusoidal tracking, Gaussian-pulse tracking, zero-crossing behavior, and VAR comparison.

Intentionally omitted:

- generated time-series `.npz` files;
- `training_results.pkl` bundles;
- repeated plots for every parameter combination;
- full result folders used during local experimentation.

Regenerate full outputs with the commands in `synthetic_stability_experiments/run_commands.txt`. Dataset 7 figures are tracked separately in `dataset7_robustness_experiments/selected_figures/`.
