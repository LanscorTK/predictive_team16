# Experiment: claude_code

## Required Input Files

The following files must be present in `../../data/` (relative to this directory):

| File | Description |
|---|---|
| `participation_2024-25_experiment_v2.tab` | UK Participation Survey 2024-25 dataset (tab-separated) |
| `participation_2024-25_data_dictionary_v2.txt` | Variable definitions and coded value descriptions |

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the notebook (optional — the notebook is already included):
   ```bash
   python3 build_notebook.py
   ```

3. Execute the notebook:
   ```bash
   jupyter nbconvert --to notebook --execute experiment_claude_code.ipynb --output experiment_claude_code.ipynb
   ```

   Or open in Jupyter and run all cells sequentially (Cell → Run All).

## Outputs

| Output | Location | Description |
|---|---|---|
| `experiment_claude_code.ipynb` | `.` | Main experiment notebook (Steps 0–5.4) |
| `run_log_claude_code.md` | `.` | Step-by-step run log with status and outputs |
| `Report_claude_code.md` | `.` | Non-technical report for policy audience |
| `best_model_name.txt` | `.` | Name of the selected final model |
| `evidence_claude_code/` | `.` | All evidence artefacts (CSVs, PNGs) |
| `baseline_lr_validation_metrics.csv` | `evidence_claude_code/` | Baseline LR validation results |
| `lr_tuning_results.csv` | `evidence_claude_code/` | LR hyperparameter tuning grid |
| `xgb_tuning_results.csv` | `evidence_claude_code/` | XGBoost hyperparameter tuning grid |
| `test_model_comparison.csv` | `evidence_claude_code/` | Final test-set comparison (3 models) |
| `model_selection_framework.csv` | `evidence_claude_code/` | Weighted scoring framework results |
| `missingness_handling_summary.csv` | `evidence_claude_code/` | Variable-level missingness strategy |
| `EDA_claude_code_Pics/` | `evidence_claude_code/` | EDA visualisations (4 PNG files) |

## Reproducibility

- **Random seed**: `random_state=42` is used throughout (numpy, random, sklearn, xgboost).
- **Relative paths**: All file paths are relative to the working directory.
- **Sequential execution**: The notebook runs top-to-bottom without manual intervention.
- **Stratified splits**: Train/validation/test split (70/15/15) preserves class balance.
- **Package versions**: Exact versions pinned in `requirements.txt`.
