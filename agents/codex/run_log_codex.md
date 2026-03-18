# Run Log: codex

## Step 0 Setup
- completion status: Completed
- key actions:
  - Reviewed variable dictionary only (no data load in this step).
  - Created notebook `experiment_codex.ipynb` with title and setup section.
  - Set reproducibility controls with fixed seed `42`.
  - Created `evidence_codex/` and EDA figure subfolder prepared by notebook.
- key outputs:
  - `experiment_codex.ipynb`
  - `evidence_codex/`
- important warnings or errors:
  - None.

## Step 1.1 Dataset Ingestion and Schema Checks
- completion status: Completed
- key actions:
  - Loaded `data/participation_2024-25_experiment_v2.tab` into `participation_raw`.
  - Checked required variable presence and obvious schema issues.
- key outputs:
  - Raw shape confirmed in notebook: `34,378 rows x 16 columns`.
  - All expected required variables present; no null-based schema break.
- important warnings or errors:
  - None.

## Step 1.2 Problem Definition
- completion status: Completed
- key actions:
  - Added task framing markdown for policy-focused under-engagement identification.
  - Declared target and features.
  - Explicitly stated `CARTS_NET` values `-3` and `3` will be dropped later.
  - Added variable table based on dictionary.
- key outputs:
  - Problem definition markdown section in notebook.
- important warnings or errors:
  - None.

## Step 2 EDA
- completion status: Completed
- key actions:
  - Dropped target-coded missing rows (`CARTS_NET` in `{-3, 3}`).
  - Built `participation_eda` by removing original `CARTS_NET` and adding binary-labelled target.
  - Produced EDA visualisations and saved PNG outputs.
- key outputs:
  - EDA sample rows after target filtering: `34,338`.
  - `evidence_codex/EDA_codex_Pics/01_target_distribution.png`
  - `evidence_codex/EDA_codex_Pics/02_ageband_target_stacked.png`
  - `evidence_codex/EDA_codex_Pics/03_education_target_stacked.png`
  - `evidence_codex/EDA_codex_Pics/04_deprivation_underengagement_line.png`
  - `evidence_codex/EDA_codex_Pics/05_wellb1_kde_by_target.png`
  - `evidence_codex/EDA_codex_Pics/06_cultsatis_distribution.png`
- important warnings or errors:
  - None.

## Step 3 Missingness Handling
- completion status: Completed
- key actions:
  - Applied variable-specific rules from coded meanings and missing rates.
  - Dropped low-rate (<5%) non-informative coded responses.
  - Re-coded high-rate coded missing values to `Unknown` for selected variables.
  - Treated `CULTSATIS=-3` as informative `RoutingSkip`.
  - Kept `EDUCAT3=-3` as informative `NoQualifications`.
  - Produced cleaned dataset `participation_clean` with binary numeric target.
- key outputs:
  - `evidence_codex/missingness_profile.csv`
  - `evidence_codex/participation_clean.csv`
  - Rows before cleaning: `34,338`
  - Rows after cleaning: `28,995`
- important warnings or errors:
  - None.

## Step 4.1 Prepare Modelling Data
- completion status: Completed
- key actions:
  - Defined `X` and `y` from `participation_clean`.
  - Built preprocessing pipelines for Logistic Regression and XGBoost.
  - Created fixed stratified train/validation/test splits (0.70/0.15/0.15).
- key outputs:
  - Split created and reused for all later tuning/evaluation cells.
- important warnings or errors:
  - None.

## Step 4.2 Create Evaluation Harness
- completion status: Completed
- key actions:
  - Defined unified evaluation metrics and threshold-search utility.
  - Prioritised F2 and recall for under-engagement detection use case.
- key outputs:
  - Reusable evaluation functions in notebook.
- important warnings or errors:
  - None.

## Step 4.3 Baseline Logistic Regression
- completion status: Completed
- key actions:
  - Trained baseline LR on train set.
  - Evaluated on validation set only with common harness.
- key outputs:
  - Baseline model metrics recorded in notebook.
- important warnings or errors:
  - None.

## Step 5.1 Improve Logistic Regression
- completion status: Completed
- key actions:
  - Tuned LR hyperparameters on validation set only.
  - Performed threshold optimisation for F2.
- key outputs:
  - `evidence_codex/lr_tuning_summary.json`
  - Best LR validation setup: `C=0.05`, `penalty=l1`, `class_weight=balanced`, `threshold=0.52`.
- important warnings or errors:
  - None.

## Step 5.2 Tune XGBoost
- completion status: Completed
- key actions:
  - Trained and tuned XGBoost on the same fixed split.
  - Performed threshold optimisation for F2.
- key outputs:
  - `evidence_codex/xgb_tuning_summary.json`
  - Best XGBoost validation setup: `n_estimators=200`, `max_depth=4`, `learning_rate=0.08`, `threshold=0.10`.
- important warnings or errors:
  - None.

## Step 5.3 Model Comparison (Test Set Only)
- completion status: Completed
- key actions:
  - Used test set only at final comparison stage.
  - Compared baseline LR, tuned LR, tuned XGBoost under identical harness.
- key outputs:
  - `evidence_codex/model_comparison_test.csv`
  - Test F2: tuned LR `0.4639`, tuned XGBoost `0.4515`, baseline LR `0.0721`.
- important warnings or errors:
  - None.

## Step 5.4 Final Model Decision
- completion status: Completed
- key actions:
  - Defined quantitative multi-metric model selection score.
  - Selected final model based on weighted policy-oriented criteria.
- key outputs:
  - `evidence_codex/final_model_selection_score.csv`
  - Final model selected: `tuned_lr` (selection score `0.4875`).
- important warnings or errors:
  - None.

## Step 6 Producing Reproducible Packaging
- completion status: Completed
- key actions:
  - Created concise dependency file and run instructions.
- key outputs:
  - `requirements.txt`
  - `README.md`
- important warnings or errors:
  - None.

## Step 7 Writing Documentation
- completion status: Completed
- key actions:
  - Produced policy-facing non-technical markdown report using actual model outputs.
- key outputs:
  - `Report_codex.md`
- important warnings or errors:
  - None.

## Notes
- The notebook initially failed execution due malformed indentation in generated cells; cells were corrected and notebook was rerun successfully end-to-end.
