# Cross-Agent Comparison: Claude Code vs Codex vs Antigravity

> **Project:** MSIN0097 Predictive Analytics — Agent Benchmarking Study
> **Generated:** 2026-03-17
> **Purpose:** Detailed comparative analysis of three AI coding agents on an end-to-end ML pipeline for arts participation under-engagement identification.

---

## 1. Data Handling

| Aspect | Claude Code | Codex | Antigravity |
|--------|-------------|-------|-------------|
| Raw rows | 34,378 | 34,338 | 34,338 |
| Strategy | Recode informative codes, drop non-informative | Variable-specific: recode high-rate, drop low-rate | Drop most invalid codes, recode CULTSATIS/EDUCAT3 |
| Rows after cleaning | **29,073** (15.3% loss) | **28,995** (15.6% loss) | **24,867** (27.6% loss) |
| CULTSATIS (-3, 71%) | Recoded to "Not asked" | Recoded to "RoutingSkip" | Retained as informative |
| Split | 70/15/15 stratified | 70/15/15 stratified | 70/15/15 stratified |

**Key insight:** Antigravity lost significantly more data (27.6% vs ~15%). Claude Code and Codex took similar tiered approaches while Antigravity was more aggressive at dropping rows.

---

## 2. EDA Quality

| Aspect | Claude Code | Codex | Antigravity |
|--------|-------------|-------|-------------|
| Plots generated | 4 (distribution, features, correlations, feature-target) | 6 (target, age, education, deprivation, wellbeing KDE, CULTSATIS) | 3 (age, education, deprivation) |
| Depth | Broad coverage + correlation heatmap | Policy-focused with KDE plots | Minimal but targeted |
| Unique contribution | Correlation analysis | Wellbeing KDE by engagement | Deprivation decile analysis |

**Winner: Codex** — most visualizations with the most policy-relevant framing.

---

## 3. Model Performance (Test Set)

| Metric | Claude Code (Tuned LR) | Codex (Tuned LR) | Antigravity (Tuned LR) |
|--------|------------------------|-------------------|------------------------|
| **Recall** | **0.769** | 0.713 | 0.735 |
| Precision | 0.170 | 0.194 | 0.208 |
| **F2 Score** | 0.452 | 0.464 | **0.488** |
| Balanced Accuracy | 0.745 | 0.745 | 0.817 |
| **ROC-AUC** | 0.829 | **0.831** | **0.853** |
| PR-AUC | 0.333 | 0.313 | 0.313 |
| Final model | Tuned LR | Tuned LR | Tuned LR |

**Key findings:**
- **All three agents selected Tuned Logistic Regression** as the final model (correcting the PROJECT_OVERVIEW which claimed Antigravity chose XGBoost).
- **Antigravity achieved the highest F2 (0.488)** and ROC-AUC (0.853) despite having the smallest training set.
- **Claude Code achieved the highest recall (0.769)** — best at catching under-engaged individuals.
- **Codex sits in the middle** with balanced metrics across the board.

---

## 4. XGBoost Results (All agents' XGB underperformed LR)

| Metric | Claude Code XGB | Codex XGB | Antigravity XGB |
|--------|----------------|-----------|-----------------|
| Recall | 0.188 | 0.620 | 0.731 |
| F2 Score | 0.212 | 0.451 | 0.427 |
| ROC-AUC | 0.796 | 0.827 | 0.812 |

Claude Code's XGBoost essentially failed on the minority class. Codex and Antigravity's XGBoost performed comparably but still lost to their tuned LR models.

---

## 5. Model Selection Framework

| Aspect | Claude Code | Codex | Antigravity |
|--------|-------------|-------|-------------|
| Approach | Weighted multi-criteria scoring | Policy-weighted multi-metric score | F2 score comparison |
| Weights | Recall 0.30, F2 0.25, BalAcc 0.20, ROC-AUC 0.15, Interp 0.10 | F2 0.45, Recall 0.20, PR-AUC 0.15, BalAcc 0.10, Prec 0.10 | Direct F2 comparison |
| Sophistication | Includes interpretability as explicit criterion | Most formalized scoring formula | Simplest approach |

**Winner: Codex** — most rigorous scoring framework with explicit formula and justification for each weight.

---

## 6. Code Quality & Engineering

| Dimension | Claude Code | Codex | Antigravity |
|-----------|-------------|-------|-------------|
| Notebook construction | Direct notebook creation | Programmatic (`build_notebook.py`) | Programmatic (`build_notebook.py`) |
| Reproducibility | seed=42 everywhere | seed=42 everywhere | seed=42 everywhere |
| Evidence trail | 2 CSVs + 4 plots | 2 CSVs + 2 JSONs + 6 plots + clean data | 3 plots + model pickle |
| LR tuning configs | 153 (9 C x 17 thresholds) | 28 (7 C x 2 pen x 2 weight) | 8 (4 C x 2 weight) |
| XGB tuning configs | 459 (27 x 17 thresholds) | 12 candidates | 12 candidates |
| Execution issues | None | Initial indentation fix | XGB deprecation warning |
| Dependencies pinned | Yes | Yes | Yes |

**Winner: Claude Code** — most thorough hyperparameter search (459 XGB configs vs 12), cleanest execution.

---

## 7. Report & Communication Quality

| Dimension | Claude Code | Codex | Antigravity |
|-----------|-------------|-------|-------------|
| Tone | Academic/policy balanced | Risk-modelling framing | Structural barriers framing |
| Policy relevance | Good — coefficients for policy interpretation | Strong — screening tool narrative | Strong — structural inequality narrative |
| Limitations acknowledged | Yes | Yes | Yes (observational, not causal) |
| Non-technical report | Present | Present | Present |

All three produced reasonable policy-facing reports. Codex and Antigravity had slightly stronger policy framing.

---

## 8. Summary Scorecard

| Criterion | Claude Code | Codex | Antigravity |
|-----------|:-----------:|:-----:|:-----------:|
| Data handling | ++ | ++ | + |
| EDA depth | ++ | +++ | + |
| Test recall | +++ | ++ | ++ |
| Test F2 | ++ | ++ | +++ |
| Test ROC-AUC | ++ | ++ | +++ |
| Model selection rigor | ++ | +++ | + |
| Tuning thoroughness | +++ | ++ | + |
| Evidence trail | ++ | +++ | + |
| Code quality | +++ | +++ | ++ |
| Report quality | ++ | ++ | ++ |

---

## 9. Critical Observations for the Group Report

1. **All agents converged on Tuned LR** — strong evidence that for this imbalanced policy problem, interpretable models with threshold tuning outperform tree-based methods.

2. **Missingness strategy directly impacted results** — Antigravity dropped 27.6% of data yet achieved the best ROC-AUC, suggesting the dropped rows may have been noisy.

3. **Threshold tuning was the key intervention** — all agents saw baseline recall ~5% jump to 65-77% purely through threshold optimization, not model complexity.

4. **XGBoost consistently underperformed** across all agents, challenging the assumption that ensemble methods always win on tabular data (likely due to severe class imbalance + limited features).

5. **Metric standardisation needed** — agents used slightly different metric definitions (macro vs minority-class PR-AUC). The comparative analysis section must normalise these.

---

## 10. Per-Agent Detailed Summaries

### 10.1 Claude Code

- **EDA:** 4 plots covering distributions, correlations, feature-target relationships. Identified 88 duplicate rows.
- **Missingness:** Tiered approach — recoded informative -3 codes (EDUCAT3, NSSEC_3, FINHARD, CINTOFT, CULTSATIS), dropped non-informative codes (-4, -5, 997, 999). 15.3% data loss.
- **Baseline LR (Validation):** Recall 0.059, F2 0.072, ROC-AUC 0.819. Poor minority detection at default threshold.
- **Tuned LR (Validation):** C=10.0, threshold=0.55. Recall 0.789, F2 0.466, ROC-AUC 0.820. Massive recall improvement.
- **XGBoost (Validation):** n_estimators=300, max_depth=7, lr=0.1, threshold=0.9. Recall 0.168, F2 0.187. Many shallow configs produced zero predictions.
- **Test (Tuned LR):** Recall 0.769, F2 0.452, ROC-AUC 0.829, Precision 0.170.
- **Selection:** Weighted scoring framework (Recall 0.30, F2 0.25, BalAcc 0.20, ROC-AUC 0.15, Interpretability 0.10). Tuned LR scored 0.717 vs XGB 0.396.
- **Strengths:** Most thorough tuning (459 XGB configs), no execution failures, excellent logging.
- **Weaknesses:** No cross-validation, XGBoost essentially failed on minority class.

### 10.2 Codex

- **EDA:** 6 plots including KDE of wellbeing by engagement status and deprivation line plot. Most policy-relevant visualisations.
- **Missingness:** Variable-specific dictionaries — recode high-rate codes (NSSEC_3, CINTOFT, CSMARTD_Count to "Unknown"), drop low-rate codes. 15.6% data loss.
- **Baseline LR (Validation):** Recall 0.059, F2 0.072, ROC-AUC 0.829. Same poor baseline as other agents.
- **Tuned LR (Validation):** C=0.05, penalty=l1, class_weight=balanced, threshold=0.52. Recall 0.781, F2 0.509, ROC-AUC 0.851.
- **XGBoost (Validation):** max_depth=4, lr=0.08, scale_pos_weight=1.0, threshold=0.10. Recall 0.719, F2 0.517, ROC-AUC 0.852.
- **Test (Tuned LR):** Recall 0.713, F2 0.464, ROC-AUC 0.831, Precision 0.194.
- **Selection:** Formal scoring formula: 0.45*F2 + 0.20*Recall + 0.15*PR_AUC + 0.10*BalAcc + 0.10*Precision. Tuned LR scored 0.488 vs XGB 0.465.
- **Strengths:** Most rigorous selection framework, richest evidence trail (CSVs + JSONs + plots + clean data export), strongest EDA.
- **Weaknesses:** Fewer tuning configurations (12 XGB candidates), initial notebook indentation bug required fix.

### 10.3 Antigravity

- **EDA:** 3 plots — under-engagement rate by age band, education level, and deprivation decile. Minimal but focused.
- **Missingness:** Dropped rows with codes [-3, -4, -5, 997, 999] for most features; retained CULTSATIS -3 and EDUCAT3 -3 as informative. 27.6% data loss (largest).
- **Baseline LR (Validation):** Recall 0.027, F2 0.033, ROC-AUC 0.809. Worst baseline performance.
- **Tuned LR (Validation):** C=1, class_weight=balanced, threshold=0.585. Recall 0.659, F2 0.438, ROC-AUC 0.813.
- **XGBoost (Validation):** max_depth=7, lr=0.1, scale_pos_weight=15.75, threshold=0.286. Recall 0.650, F2 0.373, ROC-AUC 0.748.
- **Test (Tuned LR):** Recall 0.735, F2 0.488, ROC-AUC 0.853, Precision 0.208.
- **Selection:** Direct F2 comparison — simplest approach. Tuned LR won on F2 (0.488 vs 0.427).
- **Strengths:** Best test-set F2 and ROC-AUC, programmatic notebook construction (build_notebook.py), saved model artifact (final_model.pkl).
- **Weaknesses:** Heaviest data loss, fewest tuning configs (8 LR, 12 XGB), simplest selection framework, no feature importance analysis, XGB deprecation warnings.
