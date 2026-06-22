# Synthetic Stability Experiments

This folder contains three configurable DCIts experiment pipelines:

- `hidden_driver_pipeline.py` studies observed and hidden drivers;
- `regime_change_pipeline.py` studies abrupt changes in interaction structure;
- `smooth_coefficient_pipeline.py` studies smoothly time-varying coefficients.

Each pipeline has a matching JSON configuration file and writes CSV summary tables and PDF figures.

## Run

Follow the shared setup in the [repository README](../README.md), then copy this folder into `DCIts/examples/synthetic_stability_experiments/` and copy the accompanying DCIts utility support file as described there.

Run the commands in [run_commands.txt](run_commands.txt) from the shared `workspace/` directory. It includes smoke tests, full runs, targeted sweeps, and the expected output layouts.

Use `--no-training-results` to retain tables and figures while skipping the large `training_results.pkl` bundles.
