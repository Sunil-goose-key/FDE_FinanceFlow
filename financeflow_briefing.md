# FinanceFlow Capital — AI-Assisted Underwriting Triage
## Briefing for Head of Credit Risk

### 1 — The data situation

We received 400 historical loan records from the core banking system. Of those, 30 have no credit score, 25 are missing a default outcome flag, and 15 show negative annual revenue — all likely data entry or export errors. This means only 330 records are usable for training the risk model today, and the project timeline depends on the data team recovering the missing fields from the origination system and the collections database before we can move to full production.

### 2 — The model

The system learns patterns from past loans — which borrowers repaid and which did not — and produces a risk score from 0% to 100% for each new application. It does not make a final decision; it flags applications so your analysts know where to focus. The score is based on eight key data points the credit team already reviews today, including the applicant's credit score, existing debt levels, and business revenue. The system also generates a plain-English brief explaining why each flag was raised, which the analyst can read in under 30 seconds.

### 3 — The automation

Every morning at 8:00, the following happens without anyone touching a keyboard:

1. The system pulls the latest loan applications from the overnight export file.
2. It runs a data quality check — if credit scores are missing or revenue is negative, the pipeline stops and alerts the data operations team in Slack.
3. If the data passes, the system scores every application against the risk model and assigns each one a decision: Approve, Review, or Decline.
4. All Decline and Review applications are written to a Google Sheet that serves as the analyst work queue for the day.
5. A summary message — counts of Approve, Review, and Decline decisions plus total dollar value at risk — is posted to the #credit-team Slack channel.

### 4 — Limitations and risks

First, the model is only as good as the data it was trained on. If the 330 clean records are not representative of future applicants — for example, if the bank has recently changed its lending criteria — the scores will be wrong. Second, the model can pick up hidden biases from historical lending patterns; an applicant with a low credit score who runs a healthy business could be incorrectly flagged. The analyst must always verify the numbers behind the score — check bank statements, confirm revenue, and speak to the applicant — before making a final decline decision.

### 5 — The ask

I need you to sign the attached data contract by Friday and instruct your data engineering team to prioritise the credit score backfill from the core banking system so we can complete model validation and go live on Monday 14 July.
