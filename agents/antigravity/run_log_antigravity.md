# Run Log: antigravity

## 0_setup
- **Status**: Completed
- **Key Actions**: Imported libraries, set random seed to 42, created tracking folders.
- **Key Outputs**: evidence folder, run log file.
- **Warnings/Errors**: None

## 1.1_dataset_ingestion_and_schema_checks
- **Status**: Completed
- **Key Actions**: Loaded raw data and verified presence of required variables.
- **Key Outputs**: Raw data shape: (34378, 16)
- **Warnings/Errors**: None

## 1.2_problem_defition
- **Status**: Completed
- **Key Actions**: Defined prediction task and variables in markdown.
- **Key Outputs**: None
- **Warnings/Errors**: None

## 2_EDA
- **Status**: Completed
- **Key Actions**: Created binary target, dropped missing targets. Generated and saved EDA plots.
- **Key Outputs**: EDA plots in EDA_antigravity_Pics folder.
- **Warnings/Errors**: None

## 3_missingness_handling
- **Status**: Completed
- **Key Actions**: Dropped rows with non-informative missing codes according to rules.
- **Key Outputs**: Cleaned data shape: (24867, 16)
- **Warnings/Errors**: None

## 4.1_prepare_modeling_data
- **Status**: Completed
- **Key Actions**: Split data into train, val, test and built preprocessor.
- **Key Outputs**: Train/Val/Test splits.
- **Warnings/Errors**: None

## 4.2_create_evaluation_harness
- **Status**: Completed
- **Key Actions**: Defined evaluation metrics focusing on F2 Score and PR-AUC.
- **Key Outputs**: evaluate_model function.
- **Warnings/Errors**: None

## 4.3_baseline_model_LR
- **Status**: Completed
- **Key Actions**: Trained baseline Logistic Regression model.
- **Key Outputs**: Baseline validation metrics.
- **Warnings/Errors**: None

## 5.1_improve_LR
- **Status**: Completed
- **Key Actions**: Tuned LR and found best threshold on validation set.
- **Key Outputs**: Tuned LR settings and validation metrics.
- **Warnings/Errors**: None

## 5.2_tune_XGBoost
- **Status**: Completed
- **Key Actions**: Tuned XGBoost and found best threshold on validation set.
- **Key Outputs**: Tuned XGB settings and validation metrics.
- **Warnings/Errors**: None

## 5.3_model_comparison
- **Status**: Completed
- **Key Actions**: Evaluated all three models on the test set.
- **Key Outputs**: Test set metrics.
- **Warnings/Errors**: None

## 5.4_final_model_decision
- **Status**: Completed
- **Key Actions**: Compared test metrics and selected final model.
- **Key Outputs**: Saved final_model.pkl to evidence.
- **Warnings/Errors**: None

