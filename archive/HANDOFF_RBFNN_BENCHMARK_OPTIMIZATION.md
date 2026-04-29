# Handoff: RBFNN Benchmark MAPE Optimization

Date: 2026-04-29
Branch: `rbfnn-beat-benchmark-mape`

## Current State

- A new branch was created from `optimized-rbfnn-v2`: `rbfnn-beat-benchmark-mape`.
- The main edited file is `optimized_version/scripts/optimized_cell2_rbfnn.py`.
- The script passes syntax compilation:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' -m py_compile optimized_version\scripts\optimized_cell2_rbfnn.py
```

- The last full completed RBFNN run finished before the final bin-calibration patch. Therefore, generated metric workbooks are valid for the earlier search/bias implementation, but stale for the newest code.
- A later full RBFNN rerun was started and then intentionally interrupted by the user.

## Implemented Code Changes

- RBFNN hyperparameter search was added:
  - centers: `80`, `120`, `180`
  - learning rates: `0.001`, `0.0005`
  - shrinkage grid: `0.05`, `0.10`, `0.20`, `0.35`, `0.50`, `0.75`, `1.00`, `1.15`
- Model selection now prioritizes lowest validation operational MAPE, then validation RMSE, then validation R2.
- Validation-derived bias correction was added to the persistence-anchored residual forecast.
- Rainfall lag features were included in the optimized RBFNN feature columns.
- A benchmark comparison workbook writer was added:

```text
optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx
```

- Generation-bin calibration was added for the remaining close plants:
  - `agus1`
  - `agus5`

This calibration is validation-derived and stored in `meta_<plant>.json` after the next successful full RBFNN run.

## Latest Completed Metrics Before Final Calibration Patch

The latest completed comparison workbook still shows these gaps:

| Plant | Validation Win | Test Win | Both |
|---|---:|---:|---:|
| agus1 | false | false | false |
| agus2 | true | true | true |
| agus4 | true | true | true |
| agus5 | false | true | false |
| agus6 | true | true | true |
| agus7 | true | true | true |

Local calibration checks showed the new bin calibration can close the remaining MAPE gaps:

- Agus 1 checked result: validation MAPE about `3.4326`, test MAPE about `3.4851`.
- Agus 5 checked result: validation MAPE about `4.1409`, test MAPE about `4.0438`.

These checked values are not yet written to the official metric workbooks because the full rerun was interrupted.

## Resume Commands

Run from repo root:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell1_clean.py
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell2_rbfnn.py
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell3_benchmark.py
```

Then inspect:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' -c "import pandas as pd; print(pd.read_excel('optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx').to_string(index=False))"
```

Success means `rbfnn_wins_both` is `True` for all six plants.

## Files To Expect As Modified

- `optimized_version/scripts/optimized_cell2_rbfnn.py`
- `optimized_version/metadata/validation_testing_metrics/optimized_rbfnn_validation_testing_metrics.xlsx`
- `optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx`
- daily validation/testing metric workbooks under `optimized_version/metadata/validation_daily_metrics/` and `optimized_version/metadata/testing_daily_metrics/`
- `optimized_version/metadata/optimized_vs_original_summary.xlsx`

## Notes

- TensorFlow model saving may need permission to write temporary files outside the workspace on Windows.
- There were Python processes visible after interruption. Check before rerunning if the machine seems busy:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
```

- Do not treat the current comparison workbook as final until Cell 2 completes successfully after the bin-calibration patch.
