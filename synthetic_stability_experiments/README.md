# Additional DCIts Tasks

This folder contains the final synthetic experiments:

- `hidden_driver_pipeline.py`: hidden driver and observed driver experiments;
- `regime_change_pipeline.py`: regime-change experiments;
- `smooth_coefficient_pipeline.py`: smoothly time-varying coefficient experiments.

Each pipeline has a matching JSON configuration file and writes CSV tables plus PDF figures.

Add `--no-training-results` to skip saving `training_results.pkl` bundles. Metrics and plots are still computed, but the large learned sequence files are not written to disk.

Use `run_commands.txt` for the exact command list used during development.
