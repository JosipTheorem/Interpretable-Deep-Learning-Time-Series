# Third-Party Notice

This project builds on DCIts:

https://github.com/hc-xai/dcits

DCIts is distributed under the MIT License.

The file `dcits_support/src/utils.py` is based on the DCIts utility code and includes small experiment-support changes used by the thesis pipelines:

- returning per-window `alpha`, `f`, and `C` sequences from multiple runs;
- returning MAE metrics together with MSE metrics;
- making interpretation-stability statistics safe for small numbers of runs.

Keep the original DCIts license notice when redistributing or modifying DCIts-derived files.

The experimental bead-analysis support files `experimental_bead_analysis/support_utils/src/utils_dipl.py` and `experimental_bead_analysis/support_utils/src/util_echo.py` are local thesis support code built around the DCIts API.

Eva H.'s thesis PDF and the full raw Eva bead-tracking dataset are external materials and are not redistributed in this repository. The folder `experimental_bead_analysis/sample_data/` contains a tiny 20-file `.mat` sample cluster set for reproducibility; confirm redistribution permission before publishing if needed.
