# AI Agent Benchmarking for Predictive Analytics

Benchmarking three AI agents on an end-to-end predictive analytics pipeline, as part of UCL MSIN0097 Predictive Analytics group coursework.

## Project Purpose

This project evaluates how AI coding agents perform on realistic data science tasks. Three agents — **Claude Code**, **Codex**, and **Antigravity** — each receive identical prompts from a frozen benchmark protocol and independently complete a full ML pipeline on the same dataset. Their outputs are then compared on correctness, statistical validity, reproducibility, code quality, and efficiency.

The prediction task uses the UK Participation Survey (2024–25) to classify whether respondents engaged with the arts physically in the last 12 months — framed as an under-engagement identification problem with social policy relevance.

## Repository Structure

```
PA_group/
├── README.md                 # This file
├── .gitignore
├── requirements.txt          # Shared Python dependencies
├── benchmark_notes.md        # Notes on the benchmark protocol
│
├── data/                     # Input data (shared across agents)
│   ├── participation_2024-25_experiment.tab
│   └── participation_2024-25_data_dictionary_cleaned.txt
│
├── docs/                     # Reference documents
│   ├── MSIN0097_ Predictive Analytics 25-26 Group Coursework.pdf
│   └── Pipeline_260316.docx  # Frozen benchmark protocol
│
├── agents/                   # One workspace per agent
│   ├── claude_code/
│   ├── codex/
│   └── antigravity/
│
├── outputs/                  # Cross-agent comparison outputs
└── logs/                     # Benchmark-level logs
```

Each agent works inside its own `agents/<name>/` folder. During execution, each agent will create:
- `experiment_<agent>.ipynb` — the main analysis notebook
- `run_log_<agent>.md` — step-by-step progress log
- `evidence_<agent>/` — saved outputs, figures, and artifacts
- `Report_<agent>.md` — non-technical summary report
- `requirements.txt` and `README.md` — agent-specific packaging

## Input Files

| File | Description |
|------|-------------|
| `participation_2024-25_experiment.tab` | Subset of the UK Participation Survey with 11 variables (1 target + 10 features) |
| `participation_2024-25_data_dictionary_cleaned.txt` | Variable dictionary with coded values and labels |

## Benchmark Pipeline (Steps 0–7)

The pipeline is defined in `docs/Pipeline_260316.docx`. Each agent follows the same sequence:

| Step | Task |
|------|------|
| 0 | Setup — create notebook, set seed=42, initialise logging |
| 1 | Dataset ingestion + schema checks + problem definition |
| 2 | EDA and insight generation (with plots) |
| 3 | Missingness handling |
| 4 | Baseline model training (Logistic Regression) + evaluation harness |
| 5 | Improving performance (tuning LR + XGBoost, final comparison) |
| 6 | Producing reproducible packaging (requirements.txt, README) |
| 7 | Writing documentation (non-technical report) |

## How to Run the Benchmark

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For each agent** (claude_code, codex, antigravity):
   - Navigate to the agent's workspace: `cd agents/<agent_name>/`
   - Copy or symlink the data files so they are accessible from the working directory
   - Follow the prompts in `docs/Pipeline_260316.docx` step by step (Steps 0–7)
   - Each prompt is given to the agent verbatim; no manual intervention unless errors occur

3. **After all agents complete:**
   - Compare outputs across agents using the evaluation criteria from the pipeline document
   - Save comparison tables and figures to `outputs/`

## Agents Compared

| Agent | Description |
|-------|-------------|
| Claude Code | Anthropic's CLI coding agent |
| Codex | OpenAI's coding agent |
| Antigravity | AI coding agent |

## Notes

- See `benchmark_notes.md` for protocol constraints.
- The pipeline document is the authoritative benchmark specification — do not deviate from it.
- Random seed is fixed at 42 for all agents to support reproducibility.
