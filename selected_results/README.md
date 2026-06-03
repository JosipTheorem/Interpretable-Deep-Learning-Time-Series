# Selected Results

This folder contains a compact subset of the full final-pipeline experiment output tree. It is intended for quick inspection on GitHub without storing the large generated data or training bundles.

Included here:

- summary CSV tables for all three synthetic stability tasks;
- representative hidden_driver/ comparison figures;
- regime_change/ examples for high/low SNR and source/lag/sign changes;
- smooth_coefficient/ examples showing SNR contrast, slow sinusoidal tracking, Gaussian pulse tracking, zero-crossing behavior, and VAR comparison.

Intentionally omitted:

- generated time-series `.npz` files;
- `training_results.pkl` bundles;
- repeated plots for every parameter combination;
- full result folders used during local experimentation.

The full final-pipeline outputs can be regenerated with the commands in `synthetic_stability_experiments/run_commands.txt`. Dataset 7 seminar-era figures are tracked separately in `dataset7_robustness_experiments/selected_figures/`.

