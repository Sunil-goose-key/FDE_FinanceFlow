# FinanceFlow

AI-assisted underwriting triage for FinanceFlow Capital.

## Repository contents

- `financeflow_pipeline.py`: audits historical loan data, trains the classifier, and scores daily application files
- `financeflow_model_artifact.json`: saved model artifact used by the daily scoring flow
- `financeflow_n8n_workflow.json`: n8n workflow for the weekday underwriting run
- `financeflow_handoff.md`: operating notes, environment variables, and sample results

## Local commands

Train from historical labelled data:

```powershell
python financeflow_pipeline.py train `
  --historical financeflow_raw.csv `
  --model-out financeflow_model_artifact.json `
  --clean-out financeflow_clean.csv `
  --audit-out financeflow_training_audit.json
```

Score a daily file:

```powershell
python financeflow_pipeline.py score `
  --applications financeflow_clean.csv `
  --model-in financeflow_model_artifact.json `
  --decisions-out financeflow_decisions_today.csv `
  --flagged-out financeflow_flagged_queue_today.csv `
  --summary-out financeflow_scoring_summary.json
```
