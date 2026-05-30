# DCIts Support File

Copy this file into a local DCIts clone before running the thesis pipelines:

```powershell
Copy-Item -Force .\dcits_support\src\utils.py .\DCIts\src\utils.py
```

This version of `utils.py` keeps the DCIts helper API but adds the sequence outputs and MAE values used by the analysis notebooks and pipelines.
