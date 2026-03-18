# Project Overview: AI Agent Benchmarking for Predictive Analytics

> **Module:** MSIN0097 Predictive Analytics 2025-26 (UCL)
> **Assessment:** Group Coursework — Agent Tooling for Data Science
> **Deadline:** 20/03/2026, 10:00 (Moodle submission)
> **Word limit:** 2,000 words (excl. title page, ToC, bibliography, footnotes, appendices)
> **Weighting:** 40% of total module mark
> **Teams:** 3-5 members | Anonymous submission (no names on report)

---

## 1. What This Project Is

A benchmarking study comparing three AI coding agents on a realistic end-to-end data science pipeline. Each agent receives identical prompts from a frozen protocol and independently produces a complete ML workflow. The group then writes a **single 2,000-word report** evaluating agent tooling through literature review, experiment results, and comparative analysis.

---

## 2. Deliverables (from Coursework Brief)

| # | Section | Weight | Requirements |
|---|---------|--------|--------------|
| 1 | **Literature Review** | 30% | 10+ academic papers on agentic AI/LLM tooling in DS/SE. Key themes, taxonomies (ReAct, tool calling, RAG, etc.), challenges (verification, reproducibility, failure modes). |
| 2 | **Practical Exploration & Benchmarking** | 40% | 3+ agent tools, 4+ task types benchmarked. Task specs, success criteria, evidence, failure logs, reproducibility. |
| 3 | **Comparative Analysis** | 20% | Consistent framework across agents: correctness, statistical validity, reproducibility, code quality, efficiency, safety/compliance. 1+ comparison table/figure. |
| 4 | **Reflection & Conclusion** | 10% | Key findings synthesis, best practices, "playbook" (workflow patterns, verification checklists, failure modes, when NOT to use agents). |

**Additional deliverables:** Appendices (logs, screenshots, rubrics), bibliography (10+ refs), repo link (recommended).

---

## 3. Assessment Criteria

| Criterion | Weight | What markers look for |
|-----------|--------|-----------------------|
| Depth of Literature Review | 30% | Breadth, analysis, synthesis of 10+ papers |
| Experimental Rigor | 40% | Benchmark design, task specs, tool variety, evidence credibility/reproducibility |
| Analytical Insight | 20% | Depth of comparison, failure mode identification, playbook quality |
| Clarity and Presentation | 10% | Structure, coherence, readability, proper citations |

---

## 4. Dataset & Prediction Task

- **Source:** UK Participation Survey 2024-25 (DCMS)
- **File:** `data/participation_2024-25_experiment.tab` (34,378 rows, 11 columns)
- **Dictionary:** `data/participation_2024-25_data_dictionary_cleaned.txt`
- **Target:** `CARTS_NET` — "In the last 12 months, engaged (attended OR participated) with the arts physically"
  - Values 1 (Yes) and 2 (No) kept; values -3 (Not applicable) and 3 (No & Missing) dropped
  - Binary classification: engaged vs. not engaged
- **Framing:** Under-engagement identification problem with social policy relevance

### Feature Variables (10)

| Variable | Label | Type |
|----------|-------|------|
| `AGEBAND` | Age band (16-19 to 85+) | Ordinal (15 bands) |
| `SEX` | Gender | Nominal (Female/Male) |
| `QWORK` | Employment status | Nominal (10 categories) |
| `EDUCAT3` | Highest qualification | Nominal (Degree/Other) |
| `FINHARD` | Financial hardship | Ordinal (5 levels) |
| `CINTOFT` | Internet usage frequency | Ordinal (5 levels) |
| `gor` | Region (former GOR) | Nominal (9 regions) |
| `rur11cat` | Rural/Urban | Nominal (2 levels) |
| `CHILDHH` | Children in household | Scale (0-4+) |
| `COHAB` | Living as couple | Nominal (Yes/No) |

**Key data characteristic:** Severe class imbalance (~91-93% engaged vs. 7-9% not engaged, exact ratio depends on missingness handling).

---

## 5. Benchmark Protocol (Frozen)

Defined in `docs/Pipeline_260316.docx` (also `docs/Pipeline.md`). Each agent follows Steps 0-7 identically.

| Step | Task | Key Requirements |
|------|------|-----------------|
| 0 | **Setup** | Create notebook, set seed=42, init logging, confirm files |
| 1 | **Ingestion & Schema** | Load data into `participation_raw`, verify shape/columns/types, define problem |
| 2 | **EDA** | Drop invalid target rows, create `participation_eda`, generate plots with insights |
| 3 | **Missingness** | Handle coded missing values per variable, create `participation_clean` |
| 4 | **Baseline Model** | Logistic Regression baseline, evaluation harness (imbalance-aware), validate only |
| 5 | **Improving** | Tune LR + XGBoost, compare on test set (first use), select final model with framework |
| 6 | **Packaging** | `requirements.txt` + `README.md` with run instructions |
| 7 | **Documentation** | Non-technical report (~400 words) for government arts department |

**Protocol rules:** Same prompts, same order, seed=42, relative paths, no manual intervention unless errors, document any deviations.

---

## 6. Agents Compared

| Agent | Platform | Status |
|-------|----------|--------|
| **Claude Code** | Anthropic CLI agent | All steps (0-7) COMPLETE |
| **Codex** | OpenAI coding agent | All steps (0-7) COMPLETE |
| **Antigravity** | AI coding agent | All steps (0-7) COMPLETE |

All three agents have completed the full pipeline. Artifacts are stored in `agents/<name>/`.

---

## 7. Repository Structure

```
PA_group/
├── README.md                          # Repo overview
├── requirements.txt                   # Shared Python deps
├── benchmark_notes.md                 # Protocol constraints
├── .gitignore
│
├── data/
│   ├── participation_2024-25_experiment.tab
│   └── participation_2024-25_data_dictionary_cleaned.txt
│
├── docs/
│   ├── MSIN0097_ Predictive Analytics 25-26 Group Coursework.pdf
│   ├── Pipeline_260316.docx           # Frozen benchmark protocol
│   ├── Pipeline.md                    # Markdown version of protocol
│   ├── initial_prompt                 # Initial prompt given to agents
│   ├── practical_exploration_benchmarking_structure.md
│   └── PROJECT_OVERVIEW.md            # THIS FILE
│
├── agents/
│   ├── claude_code/
│   │   ├── experiment_claude_code.ipynb
│   │   ├── run_log_claude_code.md
│   │   ├── Report_claude_code.md
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── evidence_claude_code/
│   │       ├── EDA_claude_code_Pics/ (5 plots)
│   │       ├── test_set_comparison.csv
│   │       └── model_selection_scores.csv
│   │
│   ├── codex/
│   │   ├── experiment_codex.ipynb
│   │   ├── run_log_codex.md
│   │   ├── Report_codex.md
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── evidence_codex/
│   │       ├── EDA_codex_Pics/ (3 plots)
│   │       ├── missingness_handling_summary.csv
│   │       ├── baseline_lr_validation_metrics.csv
│   │       ├── lr_tuning_results.csv
│   │       ├── xgb_tuning_results.csv
│   │       ├── test_model_comparison.csv
│   │       └── model_selection_framework.csv
│   │
│   └── antigravity/
│       ├── experiment_antigravity.ipynb
│       ├── experiment_antigravity.md
│       ├── run_log_antigravity.md
│       ├── Report_antigravity.md
│       ├── README.md
│       ├── requirements.txt
│       ├── s1_1.md ... s5_2.md (step markdown files)
│       ├── s1_1.py ... s5_1.py (step python scripts)
│       ├── add_cell.py (notebook builder helper)
│       ├── EDA_antigravity_Pics/ (3 plots)
│       └── experiment_antigravity_files/ (4 inline plots)
│
├── outputs/                           # Cross-agent comparisons (empty)
└── logs/                              # Benchmark-level logs (empty)
```

---

## 8. Agent Results Summary

### 8.1 Missingness Handling

| Aspect | Claude Code | Codex | Antigravity |
|--------|-------------|-------|-------------|
| Strategy | Tiered: drop low-missing rows, recode high-missing to "Unknown" (code 0) | Recode all invalid to "Unknown", preserve all rows | Drop ordinal/geographic invalids, recode nominals to 999 |
| Rows dropped | 4,490 (13.1%) | ~40 (target only) | Some (ordinal/geographic) |
| Final clean rows | 29,848 | 34,338 | ~33,000+ (varies) |
| COHAB handling | Recoded to "Unknown" (71% missing) | Recoded to "Unknown" | Recoded to 999 |

### 8.2 Test Set Performance (Final Models)

| Metric | Claude Code (Baseline LR) | Codex (Baseline LR) | Antigravity (Tuned XGBoost) |
|--------|--------------------------|---------------------|---------------------------|
| Accuracy | 0.715 | 0.741 | varies |
| Recall (under-engaged) | 0.693 (macro) | 0.678 | ~0.71 |
| F1 (macro) | 0.536 | 0.317 | varies |
| ROC-AUC | 0.762 | 0.786 | varies |
| PR-AUC | 0.972 (macro) | 0.287 (minority) | varies |
| Final model chosen | Baseline LR | Baseline LR | Tuned XGBoost |

**Note:** PR-AUC values differ because Claude Code reports macro-averaged PR-AUC while Codex reports minority-class PR-AUC. Direct comparison requires standardisation.

### 8.3 Final Model Selection

| Agent | Selected Model | Rationale |
|-------|---------------|-----------|
| Claude Code | Baseline LR | Weighted scoring framework; best balance of recall + interpretability |
| Codex | Baseline LR | Tuned LR identical to baseline; XGBoost had very low recall (0.077) |
| Antigravity | Tuned XGBoost | Superior non-linear capture; better recall on under-engaged group |

### 8.4 Key Differences Across Agents

| Dimension | Claude Code | Codex | Antigravity |
|-----------|-------------|-------|-------------|
| **EDA depth** | 5 plots (distributions, correlations, feature-target, missing values, model comparison) | 3 plots (target dist, feature dist, non-engagement rates) | 3 plots (target dist, age-by-target, finhard-by-target, rural-by-target) |
| **Missingness approach** | Aggressive row dropping + category recoding | Conservative — preserve all rows | Balanced — drop ordinal, recode nominal |
| **XGBoost behaviour** | Marginal improvement over LR | Very poor recall (0.077) — essentially failed on minority class | Claims strong recall (~0.71) |
| **Evaluation metrics** | Macro-averaged | Minority-class focused | Minority-class focused |
| **Notebook construction** | Direct notebook creation | Direct notebook creation | Script-based (`add_cell.py` + `.py/.md` files assembled into notebook) |
| **Evidence trail** | 2 summary CSVs + 5 plots | 6 CSVs + 3 plots | Inline notebook outputs + 3 plots |
| **Run log quality** | Detailed with metrics | Detailed with metrics | Detailed with step-by-step actions |
| **Report tone** | Academic/policy balanced | Risk-modelling framing | Policy-advocacy framing |

---

## 9. Current Project Status

### Completed

- [x] Benchmark protocol designed and frozen
- [x] Dataset prepared (subset + dictionary)
- [x] All 3 agents completed Steps 0-7
- [x] Run logs, evidence folders, reports generated for all agents
- [x] Practical exploration structure drafted (`docs/practical_exploration_benchmarking_structure.md`)

### Remaining Work for Report Submission (due 20/03/2026)

- [ ] **Literature review** — 10+ papers on agentic AI/LLM tooling (30% of mark)
- [ ] **Practical exploration section** — write up benchmark design and findings per the structure guide
- [ ] **Comparative analysis section** — consistent cross-agent evaluation framework with table/figure
- [ ] **Reflection & conclusion** — synthesis, playbook, lessons learned
- [ ] **Cross-agent comparison outputs** — populate `outputs/` with comparison tables/figures
- [ ] **Final report assembly** — 2,000-word Word/LaTeX document, executive-summary style
- [ ] **Bibliography** — proper academic referencing (APA/Harvard)
- [ ] **Appendices** — screenshots, logs, scoring rubrics, prompt records
- [ ] **Quality checks** — verify all notebooks run top-to-bottom, no fabricated results
- [ ] **Submission** — single file via Moodle

### Critical Observations for Comparative Analysis

1. **Metric inconsistency:** Claude Code uses macro-averaged metrics; Codex and Antigravity use minority-class metrics. Must standardise for fair comparison.
2. **Missingness divergence:** Agents took significantly different approaches (aggressive dropping vs. full preservation). This directly impacts training data size and potential selection bias.
3. **XGBoost performance gap:** Codex's XGBoost essentially failed to detect the minority class (recall 0.077), while Antigravity's claims strong recall. Needs verification.
4. **Antigravity's unique architecture:** Uses script-based notebook assembly (`add_cell.py`) rather than direct notebook creation — worth discussing as a different agent workflow pattern.
5. **All agents converged on class-imbalance awareness** — all used stratified splits and imbalance-aware evaluation.
6. **Two of three agents chose Logistic Regression** over XGBoost as the final model, prioritising interpretability for the policy use case.

---

## 10. Key Files Reference

| Purpose | File |
|---------|------|
| Coursework brief | `docs/MSIN0097_ Predictive Analytics 25-26 Group Coursework.pdf` |
| Frozen benchmark protocol | `docs/Pipeline_260316.docx` / `docs/Pipeline.md` |
| Initial agent prompt | `docs/initial_prompt` |
| Report structure guide | `docs/practical_exploration_benchmarking_structure.md` |
| Data dictionary | `data/participation_2024-25_data_dictionary_cleaned.txt` |
| Dataset | `data/participation_2024-25_experiment.tab` |

---

*Generated: 2026-03-16 | This document provides a complete snapshot of the project state for coordination purposes.*
