# Arts Participation Experiment (codex)

## Required Input Files
Place these files in `./data` (already expected by the notebook):
- `participation_2024-25_experiment_v2.tab`
- `participation_2024-25_data_dictionary_v2.txt`

## How To Run
From this directory (`agents/codex`):

```bash
python3 -m nbconvert --to notebook --execute experiment_codex.ipynb --output experiment_codex.ipynb
```

This executes the notebook from top to bottom with no manual intervention.

## Outputs Produced
- Main notebook (executed): `experiment_codex.ipynb`
- Run log: `run_log_codex.md`
- Evidence folder: `evidence_codex/`
  - EDA figures: `evidence_codex/EDA_codex_Pics/*.png`
  - Clean dataset: `evidence_codex/participation_clean.csv`
  - Missingness profile: `evidence_codex/missingness_profile.csv`
  - Tuning summaries: `evidence_codex/lr_tuning_summary.json`, `evidence_codex/xgb_tuning_summary.json`
  - Model comparison and final selection: `evidence_codex/model_comparison_test.csv`, `evidence_codex/final_model_selection_score.csv`
- Policy report: `Report_codex.md`

## Reproducibility Steps
- Fixed global random seed (`RANDOM_STATE = 42`) in notebook.
- Used `random_state=42` in all split/model components involving randomness.
- Reused one fixed train/validation/test split for all tuning and comparisons.
- Reserved test set for final comparison only.
