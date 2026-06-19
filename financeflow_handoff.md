# FinanceFlow Handoff

## What is in the folder

- `financeflow_pipeline.py`: single entrypoint for training and daily scoring.
- `financeflow_model_artifact.json`: trained classifier artifact from the 400-row historical export.
- `financeflow_training_audit.json`: training audit summary for the historical export.
- `financeflow_scoring_summary.json`: latest successful scoring summary.
- `financeflow_decisions_today.csv`: full scored output for the sample run.
- `financeflow_flagged_queue_today.csv`: only `Review` and `Decline` cases for analysts.
- `financeflow_n8n_workflow.json`: n8n workflow for weekday automation.

## Day 1 audit result

- Historical rows received: `400`
- Usable training rows: `330`
- Missing `credit_score`: `30`
- Missing `default_flag`: `25`
- Negative `annual_revenue_usd`: `15`
- Holdout metrics on the trained classifier:
  - Accuracy: `0.6212`
  - Precision: `0.3793`
  - Recall: `0.6111`
  - F1: `0.4681`

## Daily run commands

Train or refresh the model from a labelled historical file:

```powershell
python financeflow_pipeline.py train `
  --historical financeflow_raw.csv `
  --model-out financeflow_model_artifact.json `
  --clean-out financeflow_clean.csv `
  --audit-out financeflow_training_audit.json
```

Score a daily applications export with the saved model:

```powershell
python financeflow_pipeline.py score `
  --applications financeflow_clean.csv `
  --model-in financeflow_model_artifact.json `
  --decisions-out financeflow_decisions_today.csv `
  --flagged-out financeflow_flagged_queue_today.csv `
  --summary-out financeflow_scoring_summary.json
```

## Decision policy

- `Decline`: default probability `>= 0.65`
- `Review`: default probability `>= 0.40` and `< 0.65`
- `Approve`: default probability `< 0.40`

## n8n environment variables

- `FINANCEFLOW_RAW_CSV_URL`: internal URL for the overnight `financeflow_raw.csv` export
- `FINANCEFLOW_MODEL_PATH`: absolute path to `financeflow_model_artifact.json`
- `FINANCEFLOW_ANALYST_EMAIL`: fixed analyst mailbox for decline briefs
- `GOOGLE_SHEET_ID`: analyst queue sheet id

## Operating behavior

- Training mode removes unusable historical rows but still produces an audit summary.
- The n8n workflow follows the required business pattern: `Schedule Trigger -> HTTP Request -> Code -> If -> Google Sheets / Slack / Claude+Slack / Gmail`.
- The Code node performs the required gate checks for null `credit_score`, null `default_flag`, and negative `annual_revenue_usd`, then scores the applications with the saved model artifact.
- Helper filter nodes exist only to separate decision rows, summary rows, and decline rows so the required downstream nodes behave correctly.

## Current sample run

- Approved: `151`
- Review: `94`
- Decline: `85`
- Flagged queue size: `179`
- Total flagged loan value: `$49,533,069.04`
