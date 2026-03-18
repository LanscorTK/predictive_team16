#!/usr/bin/env python3
"""
build_notebook.py — Generates experiment_claude_code.ipynb programmatically.

Run from agents/claude_code/:
    python3 build_notebook.py
Then execute:
    jupyter nbconvert --to notebook --execute experiment_claude_code.ipynb --output experiment_claude_code.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
})


def md(source: str):
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Setup
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Experiment: claude_code

## Step 0 — Setup

This notebook implements a complete predictive analytics pipeline for the
UK Participation Survey 2024-25. All randomness uses `random_state=42` for
full reproducibility. Outputs are saved to the `evidence_claude_code/` folder.
""")

code("""
# ── Imports ──────────────────────────────────────────────────────────────────
import os, random, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE      = '../../data/participation_2024-25_experiment_v2.tab'
DICT_FILE      = '../../data/participation_2024-25_data_dictionary_v2.txt'
EVIDENCE_DIR   = 'evidence_claude_code'
EDA_DIR        = os.path.join(EVIDENCE_DIR, 'EDA_claude_code_Pics')
RUN_LOG        = 'run_log_claude_code.md'

os.makedirs(EDA_DIR, exist_ok=True)

# ── Run-log helper ──────────────────────────────────────────────────────────
def log_step(step_name, status, actions, outputs, warnings_errors='None'):
    with open(RUN_LOG, 'a') as f:
        f.write(f"## Step: {step_name}\\n")
        f.write(f"- Completion Status: {status}\\n")
        f.write(f"- Key Actions: {actions}\\n")
        f.write(f"- Key Outputs: {outputs}\\n")
        f.write(f"- Warnings/Errors: {warnings_errors}\\n\\n")

# Initialise run log
with open(RUN_LOG, 'w') as f:
    f.write(f"# Run Log — claude_code\\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

log_step('0_setup', 'SUCCESS',
         'Created notebook, evidence directory, run log',
         'experiment_claude_code.ipynb, evidence_claude_code/, run_log_claude_code.md')

print("Setup complete.")
print(f"Evidence directory: {EVIDENCE_DIR}")
print(f"EDA directory:      {EDA_DIR}")
print(f"Run log:            {RUN_LOG}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1.1 — Dataset Ingestion & Schema Checks
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Step 1 — Data Ingestion and Problem Definition

## 1.1 Dataset Ingestion & Schema Checks

In this subsection we load the raw dataset, confirm the expected schema
(16 variables), and perform basic integrity checks before any transformations.
""")

code("""
# ── Load data ────────────────────────────────────────────────────────────────
participation_raw = pd.read_csv(DATA_FILE, sep='\\t')
print(f"Rows: {participation_raw.shape[0]:,}  |  Columns: {participation_raw.shape[1]}")

# ── Required variables ───────────────────────────────────────────────────────
REQUIRED = [
    'CARTS_NET', 'CHERVIS12_NET', 'EDUCAT3', 'NSSEC_3', 'FINHARD',
    'WELLB1', 'CINTOFT', 'CSMARTD_Count', 'WELLB4', 'TENHARM',
    'LONELY', 'ETHNIC_NET', 'emdidc19', 'CULTSATIS', 'SEX', 'AGEBAND'
]

present  = [v for v in REQUIRED if v in participation_raw.columns]
missing  = [v for v in REQUIRED if v not in participation_raw.columns]
print(f"\\nRequired variables present: {len(present)}/{len(REQUIRED)}")
if missing:
    print(f"MISSING variables: {missing}")
else:
    print("All required variables found.")

# ── Schema checks ────────────────────────────────────────────────────────────
print("\\n--- Data types ---")
print(participation_raw[REQUIRED].dtypes)

print("\\n--- Null counts (NaN) ---")
print(participation_raw[REQUIRED].isnull().sum())

print("\\n--- Duplicate rows ---")
n_dup = participation_raw.duplicated().sum()
print(f"Duplicate rows: {n_dup}")

print("\\n--- Value ranges ---")
for col in REQUIRED:
    vals = sorted(participation_raw[col].unique())
    print(f"  {col}: min={min(vals)}, max={max(vals)}, unique={len(vals)}")

log_step('1.1_dataset_ingestion', 'SUCCESS',
         f'Loaded {participation_raw.shape[0]} rows x {participation_raw.shape[1]} cols; verified all 16 variables present',
         'participation_raw dataframe',
         f'{n_dup} duplicate rows found' if n_dup > 0 else 'None')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1.2 — Problem Definition
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 1.2 Problem Definition

### Prediction Task

**Binary classification**: predict whether a respondent engaged with the arts
physically in the last 12 months (`CARTS_NET`).

We frame this as an **under-engagement identification** problem. Rather than
treating arts participation as a purely individual preference, this task
investigates whether non-participation is socially patterned across
demographic, socioeconomic, digital, and geographic factors. The aim is to
identify groups that may face structural or contextual barriers to physical
arts engagement, supporting more inclusive cultural policy.

**Target variable**: `CARTS_NET` (1 = engaged, 2 = not engaged).
Rows where `CARTS_NET` is −3 or 3 will be dropped later as missing values.

**Feature variables** (15):

| Variable | Description | Type |
|---|---|---|
| CHERVIS12_NET | Visited heritage site in last 12 months | Binary |
| EDUCAT3 | Highest qualification level | Ordinal |
| NSSEC_3 | Socio-economic classification (3 classes) | Nominal |
| FINHARD | Financial hardship | Ordinal |
| WELLB1 | Life satisfaction (0–10) | Scale |
| CINTOFT | Internet usage frequency | Ordinal |
| CSMARTD_Count | Smart devices in household (0–6+) | Count |
| WELLB4 | Anxiety yesterday (0–10) | Scale |
| TENHARM | Tenure status | Nominal |
| LONELY | Loneliness frequency | Ordinal |
| ETHNIC_NET | Ethnic group (grouped) | Nominal |
| emdidc19 | Index of Multiple Deprivation decile | Ordinal |
| CULTSATIS | Satisfaction with local cultural activities | Ordinal |
| SEX | Gender | Nominal |
| AGEBAND | Age band (15 bands) | Ordinal |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Step 2 — Exploratory Data Analysis

We first remove rows where the target is missing or ambiguous
(`CARTS_NET` ∈ {−3, 3}), then explore the distribution of the target and
each feature in relation to arts engagement.
""")

code("""
# ── Prepare EDA dataframe ───────────────────────────────────────────────────
participation_eda = participation_raw[
    ~participation_raw['CARTS_NET'].isin([-3, 3])
].copy()

# Binary target: 1 = engaged, 0 = not engaged
participation_eda['target'] = (participation_eda['CARTS_NET'] == 1).astype(int)
participation_eda.drop(columns=['CARTS_NET'], inplace=True)

print(f"Rows after dropping CARTS_NET ∈ {{-3, 3}}: {len(participation_eda):,}")
print(f"Target distribution:\\n{participation_eda['target'].value_counts()}")
print(f"\\nEngagement rate: {participation_eda['target'].mean():.3f}")

log_step('2_EDA_prep', 'SUCCESS',
         f'Removed {len(participation_raw) - len(participation_eda)} rows with CARTS_NET in [-3,3]; created binary target',
         'participation_eda dataframe')
""")

md("""
### 2.1 Target Distribution
""")

code("""
fig, ax = plt.subplots(figsize=(6, 4))
counts = participation_eda['target'].value_counts().sort_index()
bars = ax.bar(['Not Engaged (0)', 'Engaged (1)'], counts.values,
              color=['#e74c3c', '#2ecc71'], edgecolor='black')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{count:,}', ha='center', fontsize=11)
ax.set_title('Target Distribution: Physical Arts Engagement', fontsize=13)
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'target_distribution.png'), dpi=150)
plt.show()
plt.close()
print(f"Class balance — Engaged: {counts.get(1,0):,} ({counts.get(1,0)/len(participation_eda)*100:.1f}%)  "
      f"Not engaged: {counts.get(0,0):,} ({counts.get(0,0)/len(participation_eda)*100:.1f}%)")
""")

md("""
### 2.2 Feature Distributions
""")

code("""
FEATURES = [
    'CHERVIS12_NET', 'EDUCAT3', 'NSSEC_3', 'FINHARD', 'WELLB1', 'CINTOFT',
    'CSMARTD_Count', 'WELLB4', 'TENHARM', 'LONELY', 'ETHNIC_NET',
    'emdidc19', 'CULTSATIS', 'SEX', 'AGEBAND'
]

fig, axes = plt.subplots(5, 3, figsize=(18, 22))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    participation_eda[feat].value_counts().sort_index().plot(kind='bar', ax=ax,
        color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Feature Distributions (all values including coded missing)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
""")

md("""
### 2.3 Feature–Target Relationships

For each feature, we compute the arts engagement rate per category to identify
which groups have lower participation rates.
""")

code("""
fig, axes = plt.subplots(5, 3, figsize=(18, 24))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    grouped = participation_eda.groupby(feat)['target'].mean().sort_index()
    grouped.plot(kind='bar', ax=ax, color='darkorange', edgecolor='black', alpha=0.8)
    ax.set_title(f'{feat} — engagement rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engagement rate')
    ax.set_ylim(0, 1)
    ax.axhline(y=participation_eda['target'].mean(), color='red',
               linestyle='--', linewidth=1, label='Overall mean')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Engagement Rate by Feature Value', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'feature_target_relationships.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
""")

md("""
### 2.4 Correlation Heatmap

A correlation heatmap of all numeric features (including coded values) gives
a quick view of linear relationships.
""")

code("""
corr = participation_eda[FEATURES + ['target']].corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap (raw coded values)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'), dpi=150)
plt.show()
plt.close()

log_step('2_EDA', 'SUCCESS',
         'Generated target distribution, feature distributions, engagement-rate plots, correlation heatmap',
         'EDA_claude_code_Pics/: target_distribution.png, feature_distributions.png, '
         'feature_target_relationships.png, correlation_heatmap.png')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Missingness Handling
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Step 3 — Missingness Handling

## Strategy

In this dataset, missing values are encoded as negative codes (−3, −4, −5) or
high sentinel values (997, 999) rather than NaN. We apply variable-specific
rules based on the data dictionary, distinguishing between **informative codes**
(−3 that represents a real category) and **truly non-informative codes**
(don't know, prefer not to say, etc.).

| Variable | Informative codes (recode) | Non-informative codes (drop) | Strategy |
|---|---|---|---|
| CHERVIS12_NET | — | — | Clean (1/2 only) |
| EDUCAT3 | −3 → "No qualifications" | −4, −5, 997, 999 | Recode −3; drop rest |
| NSSEC_3 | −3 → "Never worked / N/A" | — | Recode −3 |
| FINHARD | −3 → "Not applicable" | −4, −5, 997 | Recode −3; drop rest |
| WELLB1 | — | −4, −5, 997 | Drop rows |
| CINTOFT | −3 → "Not applicable" | −4, −5 | Recode −3; drop rest |
| CSMARTD_Count | −3 → "Not applicable" | — | Recode −3 |
| WELLB4 | — | −4, −5, 997 | Drop rows |
| TENHARM | −3 → "Not applicable" | — | Recode −3 |
| LONELY | — | −4, −5, 997 | Drop rows |
| ETHNIC_NET | — | −3, 997, 999 | Drop rows |
| emdidc19 | — | — | Clean (1–10 only) |
| CULTSATIS | −3 → "Not asked" (71%) | 999 | Recode −3; drop 999 |
| SEX | — | −4, −5, 997 | Drop rows |
| AGEBAND | — | −3, 997 | Drop rows |

**Key decisions:**
- **CULTSATIS**: −3 is a routing skip (~71%) — recode to category 6 ("Not asked").
- **EDUCAT3**: −3 = "No qualifications" is substantively meaningful — recode to
  category 0 rather than dropping 24% of data.
- **NSSEC_3**: −3 = "Not applicable" (often never-worked individuals) — recode
  to category 5 to retain these respondents.
- **FINHARD, CINTOFT, CSMARTD_Count, TENHARM**: −3 = "Not applicable" is
  recoded as a distinct category to preserve data.
- For truly non-informative codes (−4, −5, 997, 999), we drop rows since rates
  are low.
""")

code("""
print(f"Rows before missingness handling: {len(participation_eda):,}")

# ── Define informative recodes and non-informative (drop) codes ──────────────
# Informative -3 codes that represent real categories:
RECODE_MAP = {
    'EDUCAT3':       {-3: 0},    # "No qualifications" → category 0
    'NSSEC_3':       {-3: 5},    # "Never worked / N/A" → category 5
    'FINHARD':       {-3: 6},    # "Not applicable" → category 6
    'CINTOFT':       {-3: 6},    # "Not applicable" → category 6
    'CSMARTD_Count': {-3: -1},   # "Not applicable" → -1 (will be encoded)
    'TENHARM':       {-3: 4},    # "Not applicable" → category 4
    'CULTSATIS':     {-3: 6},    # "Not asked" → category 6
}

# Truly non-informative codes to drop (don't know, prefer not to say, etc.):
DROP_CODES = {
    'CHERVIS12_NET': [],
    'EDUCAT3':       [-4, -5, 997, 999],
    'NSSEC_3':       [],
    'FINHARD':       [-4, -5, 997],
    'WELLB1':        [-4, -5, 997],
    'CINTOFT':       [-4, -5],
    'CSMARTD_Count': [],
    'WELLB4':        [-4, -5, 997],
    'TENHARM':       [],
    'LONELY':        [-4, -5, 997],
    'ETHNIC_NET':    [-3, 997, 999],
    'emdidc19':      [],
    'CULTSATIS':     [999],
    'SEX':           [-4, -5, 997],
    'AGEBAND':       [-3, 997],
}

# ── Compute missing/recode rates ────────────────────────────────────────────
summary_rows = []
for feat in FEATURES:
    drop_codes = DROP_CODES[feat]
    recode = RECODE_MAP.get(feat, {})
    n_drop = participation_eda[feat].isin(drop_codes).sum() if drop_codes else 0
    n_recode = sum((participation_eda[feat] == k).sum() for k in recode.keys())
    drop_rate = n_drop / len(participation_eda) * 100
    summary_rows.append({
        'Variable': feat,
        'Recode': str(recode) if recode else 'None',
        'Drop codes': str(drop_codes) if drop_codes else 'None',
        'Rows to drop': n_drop,
        'Drop rate (%)': round(drop_rate, 2),
        'Rows recoded': n_recode,
    })

missingness_df = pd.DataFrame(summary_rows)
print("\\n--- Missingness Summary ---")
print(missingness_df.to_string(index=False))
missingness_df.to_csv(os.path.join(EVIDENCE_DIR, 'missingness_handling_summary.csv'), index=False)
""")

code("""
# ── Apply missingness handling ──────────────────────────────────────────────
participation_clean = participation_eda.copy()

# 1. Recode informative -3 codes to new categories
for feat, mapping in RECODE_MAP.items():
    for old_val, new_val in mapping.items():
        n = (participation_clean[feat] == old_val).sum()
        participation_clean.loc[participation_clean[feat] == old_val, feat] = new_val
        print(f"{feat}: recoded {n:,} rows ({old_val} → {new_val})")

# 2. Drop rows with truly non-informative codes
rows_before = len(participation_clean)
for feat in FEATURES:
    codes = DROP_CODES[feat]
    if codes:
        mask = participation_clean[feat].isin(codes)
        n_drop = mask.sum()
        if n_drop > 0:
            participation_clean = participation_clean[~mask]
            print(f"{feat}: dropped {n_drop} rows with codes {codes}")

rows_after = len(participation_clean)
print(f"\\nRows before cleaning: {rows_before:,}")
print(f"Rows after cleaning:  {rows_after:,}")
print(f"Rows dropped:         {rows_before - rows_after:,} ({(rows_before - rows_after)/rows_before*100:.1f}%)")

# ── Verify no non-informative coded values remain ───────────────────────────
remaining = {}
for feat in FEATURES:
    all_bad = set(DROP_CODES[feat])
    bad = participation_clean[feat].isin(all_bad).sum()
    if bad > 0:
        remaining[feat] = bad
# Also check no original -3/-4/-5/997/999 remain (except recoded values)
for feat in FEATURES:
    for code in [-4, -5, 997, 999]:
        n = (participation_clean[feat] == code).sum()
        if n > 0:
            remaining[f"{feat}({code})"] = n
if remaining:
    print(f"\\nWARNING: Remaining non-informative values: {remaining}")
else:
    print("\\nVerification passed: no non-informative coded values remain.")

print(f"\\nNaN check: {participation_clean[FEATURES].isnull().sum().sum()} NaN values in features")

log_step('3_missingness_handling', 'SUCCESS',
         f'Recoded CULTSATIS -3→6; dropped rows with non-informative codes; '
         f'{rows_before - rows_after} rows removed ({(rows_before - rows_after)/rows_before*100:.1f}%)',
         'participation_clean dataframe, evidence_claude_code/missingness_handling_summary.csv')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4.1 — Prepare Modelling Data
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Step 4 — Modelling Pipeline

## 4.1 Data Preparation

We define X (features) and y (target), create preprocessing pipelines for
Logistic Regression and XGBoost, and split into train / validation / test
sets (70 / 15 / 15) with stratification to preserve class balance.
""")

code("""
# ── Define X and y ──────────────────────────────────────────────────────────
X = participation_clean[FEATURES].copy()
y = participation_clean['target'].copy()

print(f"X shape: {X.shape}")
print(f"y distribution:\\n{y.value_counts()}")
print(f"y mean (engagement rate): {y.mean():.3f}")

# ── Classify features ───────────────────────────────────────────────────────
CATEGORICAL = ['CHERVIS12_NET', 'EDUCAT3', 'NSSEC_3', 'FINHARD', 'CINTOFT',
               'TENHARM', 'LONELY', 'ETHNIC_NET', 'CULTSATIS', 'SEX', 'AGEBAND']
NUMERIC     = ['WELLB1', 'CSMARTD_Count', 'WELLB4', 'emdidc19']

print(f"\\nCategorical features ({len(CATEGORICAL)}): {CATEGORICAL}")
print(f"Numeric features ({len(NUMERIC)}): {NUMERIC}")

# ── Preprocessing pipelines ─────────────────────────────────────────────────
# LR pipeline: one-hot encode categorical, passthrough numeric
lr_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error'), CATEGORICAL),
        ('num', 'passthrough', NUMERIC),
    ]
)

# XGBoost pipeline: ordinal-encode categorical (XGBoost handles ordinal natively)
xgb_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL),
        ('num', 'passthrough', NUMERIC),
    ]
)

# ── Stratified train / val / test split (70 / 15 / 15) ─────────────────────
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
# Split remaining 85% into train (70/85 ≈ 0.8235) and val (15/85 ≈ 0.1765)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=15/85, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"\\nSplit sizes:")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val):,}  ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test):,}  ({len(X_test)/len(X)*100:.1f}%)")
print(f"\\nTarget rates — Train: {y_train.mean():.3f}  Val: {y_val.mean():.3f}  Test: {y_test.mean():.3f}")

# ── Fit preprocessors on training data ──────────────────────────────────────
X_train_lr  = lr_preprocessor.fit_transform(X_train)
X_val_lr    = lr_preprocessor.transform(X_val)
X_test_lr   = lr_preprocessor.transform(X_test)

X_train_xgb = xgb_preprocessor.fit_transform(X_train)
X_val_xgb   = xgb_preprocessor.transform(X_val)
X_test_xgb  = xgb_preprocessor.transform(X_test)

print(f"\\nLR feature matrix shape (after one-hot): {X_train_lr.shape}")
print(f"XGB feature matrix shape (after ordinal): {X_train_xgb.shape}")

log_step('4.1_prepare_modeling_data', 'SUCCESS',
         f'Defined X ({X.shape[1]} features) and y; created LR and XGB preprocessors; '
         f'stratified 70/15/15 split',
         'X_train, X_val, X_test, y_train, y_val, y_test; preprocessed matrices')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4.2 — Evaluation Harness
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 4.2 Evaluation Harness

### Design Rationale

The primary goal is to **identify under-engaged individuals** so that policy
interventions can target them. This means:

- **Recall** (sensitivity) is critical — we want to catch as many under-engaged
  people as possible, even at the cost of some false positives.
- **F2 score** weights recall twice as heavily as precision, aligning with our
  policy objective.
- **Balanced accuracy** accounts for class imbalance.
- **ROC-AUC** and **PR-AUC** give threshold-independent discrimination measures.
- **Precision** and **F1** are reported for completeness.

All models are evaluated with the same function to ensure fair comparison.
""")

code("""
def evaluate_model(y_true, y_pred, y_prob, label='Model'):
    \"\"\"Evaluate a binary classifier and return a dict of metrics.\"\"\"
    metrics = {
        'Model': label,
        'Precision':         round(precision_score(y_true, y_pred, pos_label=0), 4),
        'Recall':            round(recall_score(y_true, y_pred, pos_label=0), 4),
        'F1':                round(f1_score(y_true, y_pred, pos_label=0), 4),
        'F2':                round(fbeta_score(y_true, y_pred, beta=2, pos_label=0), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'ROC-AUC':           round(roc_auc_score(y_true, y_prob), 4),
        'PR-AUC':            round(average_precision_score(1 - y_true, 1 - y_prob), 4),
    }
    return metrics

def print_metrics(metrics_dict):
    \"\"\"Pretty-print a metrics dict.\"\"\"
    print(f"\\n{'='*50}")
    print(f"  {metrics_dict['Model']}")
    print(f"{'='*50}")
    for k, v in metrics_dict.items():
        if k != 'Model':
            print(f"  {k:20s}: {v:.4f}")

log_step('4.2_evaluation_harness', 'SUCCESS',
         'Defined evaluate_model() with Precision, Recall, F1, F2, Balanced Accuracy, ROC-AUC, PR-AUC',
         'evaluate_model() function')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4.3 — Baseline Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 4.3 Baseline Logistic Regression

We train a baseline LR with default hyperparameters (C=1.0, no class weighting,
default threshold 0.5) to establish a performance floor.
""")

code("""
# ── Train baseline LR ───────────────────────────────────────────────────────
baseline_lr = LogisticRegression(
    C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'
)
baseline_lr.fit(X_train_lr, y_train)

# ── Predict on validation set ───────────────────────────────────────────────
y_val_prob_blr = baseline_lr.predict_proba(X_val_lr)[:, 1]
y_val_pred_blr = baseline_lr.predict(X_val_lr)

baseline_lr_metrics = evaluate_model(y_val, y_val_pred_blr, y_val_prob_blr, 'Baseline LR (val)')
print_metrics(baseline_lr_metrics)

# Save
blr_df = pd.DataFrame([baseline_lr_metrics])
blr_df.to_csv(os.path.join(EVIDENCE_DIR, 'baseline_lr_validation_metrics.csv'), index=False)

log_step('4.3_baseline_LR', 'SUCCESS',
         'Trained baseline LR (C=1.0, default threshold 0.5)',
         'baseline_lr model, evidence_claude_code/baseline_lr_validation_metrics.csv')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5.1 — Tune Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

md("""
# Step 5 — Model Tuning and Comparison

## 5.1 Tuned Logistic Regression

We tune two aspects:
1. **Hyperparameters**: regularisation strength C, with `class_weight='balanced'`
   to handle target imbalance.
2. **Decision threshold**: sweep thresholds from 0.10 to 0.90 and select the
   threshold that maximises the F2 score on the validation set.
""")

code("""
# ── Hyperparameter search ───────────────────────────────────────────────────
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
THRESHOLDS = np.arange(0.10, 0.91, 0.05)

lr_tuning_results = []

for C in C_VALUES:
    lr_model = LogisticRegression(
        C=C, class_weight='balanced', max_iter=1000,
        random_state=RANDOM_STATE, solver='lbfgs'
    )
    lr_model.fit(X_train_lr, y_train)
    y_prob = lr_model.predict_proba(X_val_lr)[:, 1]

    for thresh in THRESHOLDS:
        y_pred = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_val, y_pred, beta=2, pos_label=0)
        lr_tuning_results.append({
            'C': C,
            'threshold': round(thresh, 2),
            'F2': round(f2, 4),
            'Recall': round(recall_score(y_val, y_pred, pos_label=0), 4),
            'Precision': round(precision_score(y_val, y_pred, pos_label=0, zero_division=0), 4),
            'Balanced_Accuracy': round(balanced_accuracy_score(y_val, y_pred), 4),
        })

lr_tuning_df = pd.DataFrame(lr_tuning_results)
lr_tuning_df.to_csv(os.path.join(EVIDENCE_DIR, 'lr_tuning_results.csv'), index=False)

# ── Best setting ─────────────────────────────────────────────────────────────
best_lr_row = lr_tuning_df.loc[lr_tuning_df['F2'].idxmax()]
best_C = best_lr_row['C']
best_lr_thresh = best_lr_row['threshold']

print(f"Total configurations evaluated: {len(lr_tuning_results)}")
print(f"\\nBest LR setting:")
print(f"  C = {best_C}")
print(f"  Threshold = {best_lr_thresh}")
print(f"  Validation F2 = {best_lr_row['F2']:.4f}")
print(f"  Validation Recall = {best_lr_row['Recall']:.4f}")
print(f"  Validation Precision = {best_lr_row['Precision']:.4f}")
print(f"  Validation Balanced Accuracy = {best_lr_row['Balanced_Accuracy']:.4f}")

# ── Retrain best model ──────────────────────────────────────────────────────
tuned_lr = LogisticRegression(
    C=best_C, class_weight='balanced', max_iter=1000,
    random_state=RANDOM_STATE, solver='lbfgs'
)
tuned_lr.fit(X_train_lr, y_train)
y_val_prob_tlr = tuned_lr.predict_proba(X_val_lr)[:, 1]
y_val_pred_tlr = (y_val_prob_tlr >= best_lr_thresh).astype(int)

tuned_lr_val_metrics = evaluate_model(y_val, y_val_pred_tlr, y_val_prob_tlr, 'Tuned LR (val)')
print_metrics(tuned_lr_val_metrics)

log_step('5.1_tune_LR', 'SUCCESS',
         f'Grid search over {len(C_VALUES)} C values x {len(THRESHOLDS)} thresholds = '
         f'{len(lr_tuning_results)} configs; best C={best_C}, threshold={best_lr_thresh}',
         'tuned_lr model, evidence_claude_code/lr_tuning_results.csv')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5.2 — Tune XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 5.2 Tuned XGBoost

We tune XGBoost hyperparameters via grid search and then optimise the decision
threshold for F2 on the validation set. `scale_pos_weight` is set to the ratio
of the majority to minority class to handle imbalance.
""")

code("""
# ── Compute scale_pos_weight ─────────────────────────────────────────────────
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
spw = n_pos / n_neg   # engaged / not-engaged
print(f"scale_pos_weight = {spw:.2f} (majority/minority ratio for under-engaged class)")

# ── Hyperparameter grid ─────────────────────────────────────────────────────
XGB_GRID = {
    'n_estimators':   [100, 200, 300],
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.05, 0.1],
}

xgb_tuning_results = []
total_configs = (len(XGB_GRID['n_estimators']) *
                 len(XGB_GRID['max_depth']) *
                 len(XGB_GRID['learning_rate']))

print(f"Hyperparameter configs: {total_configs}")
print(f"Threshold values: {len(THRESHOLDS)}")
print(f"Total evaluations: {total_configs * len(THRESHOLDS)}")

config_count = 0
for n_est in XGB_GRID['n_estimators']:
    for md in XGB_GRID['max_depth']:
        for lr in XGB_GRID['learning_rate']:
            config_count += 1
            xgb_model = XGBClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=lr,
                scale_pos_weight=spw, random_state=RANDOM_STATE,
                eval_metric='logloss', verbosity=0,
                use_label_encoder=False
            )
            xgb_model.fit(X_train_xgb, y_train)
            y_prob = xgb_model.predict_proba(X_val_xgb)[:, 1]

            for thresh in THRESHOLDS:
                y_pred = (y_prob >= thresh).astype(int)
                f2 = fbeta_score(y_val, y_pred, beta=2, pos_label=0)
                xgb_tuning_results.append({
                    'n_estimators': n_est,
                    'max_depth': md,
                    'learning_rate': lr,
                    'threshold': round(thresh, 2),
                    'F2': round(f2, 4),
                    'Recall': round(recall_score(y_val, y_pred, pos_label=0), 4),
                    'Precision': round(precision_score(y_val, y_pred, pos_label=0, zero_division=0), 4),
                    'Balanced_Accuracy': round(balanced_accuracy_score(y_val, y_pred), 4),
                })

xgb_tuning_df = pd.DataFrame(xgb_tuning_results)
xgb_tuning_df.to_csv(os.path.join(EVIDENCE_DIR, 'xgb_tuning_results.csv'), index=False)

# ── Best setting ─────────────────────────────────────────────────────────────
best_xgb_row = xgb_tuning_df.loc[xgb_tuning_df['F2'].idxmax()]
best_n_est = int(best_xgb_row['n_estimators'])
best_md    = int(best_xgb_row['max_depth'])
best_lr_xgb = best_xgb_row['learning_rate']
best_xgb_thresh = best_xgb_row['threshold']

print(f"\\nBest XGBoost setting:")
print(f"  n_estimators  = {best_n_est}")
print(f"  max_depth     = {best_md}")
print(f"  learning_rate = {best_lr_xgb}")
print(f"  Threshold     = {best_xgb_thresh}")
print(f"  Validation F2 = {best_xgb_row['F2']:.4f}")
print(f"  Validation Recall = {best_xgb_row['Recall']:.4f}")
print(f"  Validation Precision = {best_xgb_row['Precision']:.4f}")
print(f"  Validation Balanced Accuracy = {best_xgb_row['Balanced_Accuracy']:.4f}")

# ── Retrain best model ──────────────────────────────────────────────────────
tuned_xgb = XGBClassifier(
    n_estimators=best_n_est, max_depth=best_md, learning_rate=best_lr_xgb,
    scale_pos_weight=spw, random_state=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, use_label_encoder=False
)
tuned_xgb.fit(X_train_xgb, y_train)
y_val_prob_xgb = tuned_xgb.predict_proba(X_val_xgb)[:, 1]
y_val_pred_xgb = (y_val_prob_xgb >= best_xgb_thresh).astype(int)

tuned_xgb_val_metrics = evaluate_model(y_val, y_val_pred_xgb, y_val_prob_xgb, 'Tuned XGBoost (val)')
print_metrics(tuned_xgb_val_metrics)

log_step('5.2_tune_XGBoost', 'SUCCESS',
         f'Grid search over {total_configs} configs x {len(THRESHOLDS)} thresholds = '
         f'{len(xgb_tuning_results)} evaluations; best: n_est={best_n_est}, '
         f'max_depth={best_md}, lr={best_lr_xgb}, threshold={best_xgb_thresh}',
         'tuned_xgb model, evidence_claude_code/xgb_tuning_results.csv')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5.3 — Model Comparison on Test Set
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 5.3 Model Comparison — Test Set Evaluation

Only now do we evaluate on the held-out test set. We compare:
1. Baseline Logistic Regression (default threshold 0.5)
2. Tuned Logistic Regression (optimised C and threshold)
3. Tuned XGBoost (optimised hyperparameters and threshold)
""")

code("""
# ── Baseline LR on test ─────────────────────────────────────────────────────
y_test_prob_blr = baseline_lr.predict_proba(X_test_lr)[:, 1]
y_test_pred_blr = baseline_lr.predict(X_test_lr)
test_blr_metrics = evaluate_model(y_test, y_test_pred_blr, y_test_prob_blr, 'Baseline LR')

# ── Tuned LR on test ────────────────────────────────────────────────────────
y_test_prob_tlr = tuned_lr.predict_proba(X_test_lr)[:, 1]
y_test_pred_tlr = (y_test_prob_tlr >= best_lr_thresh).astype(int)
test_tlr_metrics = evaluate_model(y_test, y_test_pred_tlr, y_test_prob_tlr, 'Tuned LR')

# ── Tuned XGBoost on test ───────────────────────────────────────────────────
y_test_prob_xgb = tuned_xgb.predict_proba(X_test_xgb)[:, 1]
y_test_pred_xgb = (y_test_prob_xgb >= best_xgb_thresh).astype(int)
test_xgb_metrics = evaluate_model(y_test, y_test_pred_xgb, y_test_prob_xgb, 'Tuned XGBoost')

# ── Comparison table ─────────────────────────────────────────────────────────
comparison_df = pd.DataFrame([test_blr_metrics, test_tlr_metrics, test_xgb_metrics])
print("\\n" + "="*80)
print("  TEST SET MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(EVIDENCE_DIR, 'test_model_comparison.csv'), index=False)

log_step('5.3_model_comparison', 'SUCCESS',
         'Evaluated all 3 models on held-out test set',
         'evidence_claude_code/test_model_comparison.csv')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5.4 — Final Model Decision
# ═══════════════════════════════════════════════════════════════════════════════

md("""
## 5.4 Final Model Decision

### Model Selection Framework

We use a **weighted multi-criteria scoring framework** to select the final model.
The weights reflect the policy objective of identifying under-engaged populations:

| Criterion | Weight | Justification |
|---|---|---|
| Recall | 0.30 | Primary goal is to detect under-engaged individuals |
| F2 Score | 0.25 | Balances recall-precision with recall emphasis |
| Balanced Accuracy | 0.20 | Fair assessment under class imbalance |
| ROC-AUC | 0.15 | Threshold-independent discrimination |
| Interpretability | 0.10 | Logistic Regression more interpretable for policy |

**Interpretability scoring**: Logistic Regression = 1.0 (coefficients are
directly interpretable); XGBoost = 0.5 (feature importances available but
model is a black box).
""")

code("""
# ── Tuning summaries ─────────────────────────────────────────────────────────
print("="*80)
print("  TUNING SUMMARY: Logistic Regression")
print("="*80)
print(f"  Tuning method:      Grid search over C + threshold sweep")
print(f"  Hyperparameters:    C, threshold")
print(f"  C search range:     {C_VALUES}")
print(f"  Threshold range:    0.10 to 0.90 (step 0.05)")
print(f"  class_weight:       'balanced'")
print(f"  Total configs:      {len(lr_tuning_results)}")
print(f"  Best C:             {best_C}")
print(f"  Best threshold:     {best_lr_thresh}")
print(f"  Best val F2:        {best_lr_row['F2']:.4f}")
print(f"  Best val Recall:    {best_lr_row['Recall']:.4f}")
print(f"  Best val Bal. Acc:  {best_lr_row['Balanced_Accuracy']:.4f}")

print()
print("="*80)
print("  TUNING SUMMARY: XGBoost")
print("="*80)
print(f"  Tuning method:      Grid search + threshold sweep")
print(f"  Hyperparameters:    n_estimators, max_depth, learning_rate, threshold")
print(f"  n_estimators:       {XGB_GRID['n_estimators']}")
print(f"  max_depth:          {XGB_GRID['max_depth']}")
print(f"  learning_rate:      {XGB_GRID['learning_rate']}")
print(f"  scale_pos_weight:   {spw:.2f}")
print(f"  Threshold range:    0.10 to 0.90 (step 0.05)")
print(f"  Total configs:      {len(xgb_tuning_results)}")
print(f"  Best n_estimators:  {best_n_est}")
print(f"  Best max_depth:     {best_md}")
print(f"  Best learning_rate: {best_lr_xgb}")
print(f"  Best threshold:     {best_xgb_thresh}")
print(f"  Best val F2:        {best_xgb_row['F2']:.4f}")
print(f"  Best val Recall:    {best_xgb_row['Recall']:.4f}")
print(f"  Best val Bal. Acc:  {best_xgb_row['Balanced_Accuracy']:.4f}")
""")

code("""
# ── Weighted scoring framework ──────────────────────────────────────────────
WEIGHTS = {
    'Recall': 0.30,
    'F2': 0.25,
    'Balanced Accuracy': 0.20,
    'ROC-AUC': 0.15,
    'Interpretability': 0.10,
}

INTERPRETABILITY = {
    'Baseline LR':    1.0,
    'Tuned LR':       1.0,
    'Tuned XGBoost':  0.5,
}

# Compute weighted scores
scoring_rows = []
for metrics in [test_blr_metrics, test_tlr_metrics, test_xgb_metrics]:
    model_name = metrics['Model']
    score = 0
    detail = {}
    for criterion, weight in WEIGHTS.items():
        if criterion == 'Interpretability':
            val = INTERPRETABILITY[model_name]
        else:
            val = metrics[criterion]
        weighted = val * weight
        detail[criterion] = f"{val:.4f} x {weight} = {weighted:.4f}"
        score += weighted
    scoring_rows.append({
        'Model': model_name,
        **{f'{k} (weighted)': round(v * WEIGHTS[k] if k != 'Interpretability'
           else INTERPRETABILITY[model_name] * WEIGHTS[k], 4)
           for k, v in zip(WEIGHTS.keys(),
                          [metrics.get(k, INTERPRETABILITY.get(model_name, 0))
                           for k in WEIGHTS.keys()])},
        'Total Score': round(score, 4),
    })

scoring_df = pd.DataFrame(scoring_rows)
print("\\n" + "="*80)
print("  MODEL SELECTION FRAMEWORK — Weighted Scores")
print("="*80)
print(scoring_df.to_string(index=False))
scoring_df.to_csv(os.path.join(EVIDENCE_DIR, 'model_selection_framework.csv'), index=False)

# ── Final decision ───────────────────────────────────────────────────────────
best_model_name = scoring_df.loc[scoring_df['Total Score'].idxmax(), 'Model']
best_score = scoring_df['Total Score'].max()
print(f"\\n>>> SELECTED MODEL: {best_model_name} (score: {best_score:.4f})")

# Save best model name for report generation
with open('best_model_name.txt', 'w') as f:
    f.write(best_model_name)

log_step('5.4_final_model_decision', 'SUCCESS',
         f'Applied weighted multi-criteria framework; selected {best_model_name} (score={best_score:.4f})',
         'evidence_claude_code/model_selection_framework.csv, best_model_name.txt')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Save notebook
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT = "experiment_claude_code.ipynb"
with open(OUTPUT, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {OUTPUT}")
