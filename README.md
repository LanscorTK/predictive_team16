# Agent Tooling for Data Science — Benchmarking Repository

**Module:** MSIN0097 Predictive Analytics 2025–26 (UCL)
**Assessment:** Group Coursework — Team 16

## Overview

This repository contains the benchmarking harness and agent outputs for a study comparing three AI coding agents on an end-to-end data science pipeline. Each agent received identical prompts and independently completed a full ML workflow — from data ingestion through model selection to reproducible packaging — on the UK Participation Survey 2024–25.

**Agents compared:** Claude Code (Anthropic), Codex (OpenAI), Antigravity

**Prediction task:** Binary classification of arts engagement (under-engagement identification), using 15 demographic, socioeconomic, and geographic features.

## Repository Structure

```
├── data/                          # Shared input data
│   ├── participation_2024-25_experiment.tab
│   └── participation_2024-25_data_dictionary_cleaned.txt
│
├── agents/
│   ├── prompts/                   # Frozen prompt protocol (14 steps)
│   ├── claude_code/               # Agent workspace + evidence
│   ├── codex/                     # Agent workspace + evidence
│   └── antigravity/               # Agent workspace + evidence
│
├── requirements.txt               # Shared Python dependencies
└── .gitignore
```

Each agent workspace contains:
- `experiment_<agent>.ipynb` — main analysis notebook
- `build_notebook.py` — notebook construction script
- `run_log_<agent>.md` — step-by-step execution log
- `evidence_<agent>/` — saved metrics (CSVs/JSONs), EDA plots (PNGs)
- `Report_<agent>.md` — non-technical policy report
- `README.md` and `requirements.txt` — agent-specific reproducibility docs

## Benchmark Protocol

All agents followed a fixed 8-step pipeline (see `agents/prompts/`):

| Step | Task |
|------|------|
| 0 | Setup: notebook, seed=42, logging |
| 1 | Dataset ingestion, schema checks, problem definition |
| 2 | Exploratory data analysis with visualisations |
| 3 | Missingness handling |
| 4 | Baseline logistic regression + evaluation harness |
| 5 | Hyperparameter tuning (LR + XGBoost), model comparison on test set |
| 6 | Reproducible packaging |
| 7 | Non-technical documentation |

Prompts were delivered in fixed order with no agent-specific wording. Manual intervention was restricted to recovering from execution failures.

## How Prompts Were Used

Each agent was first given the following initialisation instruction:

> Act as an SDE to perform a project with me. Your name is [agent_name], so whenever you see [agent_name] in any files you have access to, remember to change that to your name. Your work should only be done in ./agents/[agent_name] and NOWHERE ELSE. Now implement step-by-step following the prompts in ./agents/prompts.

The agent then received the 14 prompt files from `agents/prompts/` one at a time, in filename order (`0_setup`, `1.1_dataset_ingestion_and_schema_checks`, ..., `7_writing_documentation`). Each prompt was copied verbatim into the agent's chat interface. The agent read the prompt, executed the task in its workspace, and the next prompt was sent only after the previous step completed. The `[agent_name]` placeholder in prompts was automatically resolved by the agent based on the initialisation instruction.

## Reproducing Agent Runs

```bash
pip install -r requirements.txt
cd agents/<agent_name>/
jupyter nbconvert --to notebook --execute experiment_<agent_name>.ipynb
```

All notebooks use `random_state=42` and relative paths. See each agent's `README.md` for agent-specific instructions.
