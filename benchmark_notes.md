# Benchmark Protocol Notes

## Frozen Protocol

The file `docs/Pipeline_260316.docx` is the **frozen benchmark protocol** for this project. It defines the complete end-to-end pipeline that each AI agent must follow.

## Key Rules

- The protocol defines **Steps 0–7**, each with an exact prompt, objective success criteria, and quality scoring rubrics.
- Every agent receives the **same prompts** in the **same order** to ensure a fair comparison.
- Do **not** deviate from the protocol unless an error makes continuation impossible. If intervention is required, document it clearly in the agent's run log.
- The prompts are designed to balance specificity (so agents can complete each step) with openness (so differences in agent behaviour are visible).

## Evaluation

Each step has two types of success criteria:

1. **Objective criteria** — binary checks on whether required outputs exist and are correct.
2. **Quality criteria** — rubric-scored (1–5) assessments of implementation quality, covering areas such as clarity, reproducibility, statistical rigour, and notebook communication.

Quality criteria require human judgement and will be assessed during the comparative analysis phase of the report.

## Important Reminders

- Random seed = 42 for all agents.
- All file paths must be relative.
- Notebooks must run sequentially from top to bottom without manual intervention.
- Do not fabricate results, runs, or citations.
- Keep an audit trail of all agent interactions (logs, screenshots, evidence folders).
