# Predictive Analytics Experiment

This repository contains the predictive modeling analysis for arts engagement using data from the UK Participation Survey 2024-25.

## Required Input Files
- Jupyter Notebook: `experiment_antigravity.ipynb`
- Data folder containing:
  - `participation_2024-25_experiment_v2.tab`: The main training dataset.
  - `participation_2024-25_data_dictionary_v2.txt`: Dictionary explaining the columns.
  
Note: The notebook reads data from `./data/participation_2024-25_experiment_v2.tab`.

## How to Run
Ensure you have installed the packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Run the notebook sequentially from top to bottom, either using a GUI (Jupyter/VSCode) or running nbconvert:
```bash
jupyter nbconvert --to notebook --execute --inplace experiment_antigravity.ipynb
```

## Outputs Produced
1. **`evidence_antigravity/`**: Used to save the EDA images and model exports.
    - `EDA_antigravity_Pics/*.png`: Charts generated during exploratory data analysis.
    - `final_model.pkl`: Best performing model (Tuned Logistic Regression object).
2. **`run_log_antigravity.md`**: Provides a step-by-step trace of actions, data shapes, and metrics resulting from pipeline execution.
3. **`Report_antigravity.md`**: Non-technical report containing the summary of methods and results for policy stakeholders.

## Reproducibility
- Features extensive markdown logs step-by-step.
- A fixed global random seed (`np.random.seed(42)` and `random_state=42`) was utilized during `train_test_split`, as well as within all model instances to ensure that the exact splits, tuning processes, and resulting performances remain exactly identical across separate runs.
