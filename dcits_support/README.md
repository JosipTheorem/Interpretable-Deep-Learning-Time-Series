# DCIts Utility Support

This folder is used automatically by the synthetic stability pipelines. Keep it inside this repository; do not copy it into the upstream DCIts checkout.

Its `utils.py` module keeps the DCIts helper API while adding the per-window sequence outputs and MAE values used by the synthetic analyses. At runtime, the pipelines combine this local module with `src.dcits` from the sibling DCIts checkout without modifying either source tree.
