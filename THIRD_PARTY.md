# Third-Party Notice

This project builds on DCIts:

https://github.com/hc-xai/dcits

DCIts is distributed under the MIT License.

The file `dcits_support/src/utils.py` is based on the DCIts utility code and includes small experiment-support changes used by the thesis pipelines:

- returning per-window `alpha`, `f`, and `C` sequences from multiple runs;
- returning MAE metrics together with MSE metrics;
- making interpretation-stability statistics safe for small numbers of runs.

Keep the original DCIts license notice when redistributing or modifying DCIts-derived files.
