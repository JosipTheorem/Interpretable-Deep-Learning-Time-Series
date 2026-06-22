# Third-Party Notice

This project builds on [DCIts](https://github.com/hc-xai/dcits), which is distributed under the MIT License.

The file `dcits_support/src/utils.py` is based on DCIts utility code and includes experiment-support changes used by the synthetic pipelines:

- returning per-window `alpha`, `f`, and `C` sequences from multiple runs;
- returning MAE metrics together with MSE metrics;
- making interpretation-stability statistics safe for small numbers of runs.

Keep the original DCIts license notice when redistributing or modifying DCIts-derived files.

The experimental bead-analysis support files `experimental_bead_analysis/support_utils/src/utils_dipl.py` and `experimental_bead_analysis/support_utils/src/util_echo.py` are local support code built around the DCIts API.

The full raw particle-tracking data set is external and is not redistributed in this repository. The folder `experimental_bead_analysis/sample_data/` contains a compact `.mat` sample cluster set for reproducibility.
