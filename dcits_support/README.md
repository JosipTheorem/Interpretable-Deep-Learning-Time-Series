# DCIts Support File

Copy this file into a local DCIts clone before running the final thesis pipelines. From the parent workspace that contains both repositories, use:

```powershell
Copy-Item -Force .\Interpretable-Deep-Learning-Time-Series\dcits_support\src\utils.py .\DCIts\src\utils.py
```

From inside this repository, adjust the destination path to point at your local `DCIts/src/utils.py`.

This version of `utils.py` keeps the DCIts helper API but adds the sequence outputs and MAE values used by the analysis notebooks and pipelines.
