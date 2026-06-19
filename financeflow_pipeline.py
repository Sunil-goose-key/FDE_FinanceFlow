#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import Counter


NUMERIC_FIELDS = [
    "business_age_years",
    "num_employees",
    "annual_revenue_usd",
    "loan_amount_usd",
    "loan_term_months",
    "interest_rate_pct",
    "credit_score",
    "debt_to_income_ratio",
]

CATEGORICAL_FIELDS = ["sector", "loan_purpose", "state"]
TRAINING_REQUIRED_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS + ["loan_id", "default_flag"]
SCORING_REQUIRED_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS + ["loan_id"]
ALLOWED_SECTORS = {
    "Retail",
    "Manufacturing",
    "Technology",
    "Hospitality",
    "Healthcare",
    "Construction",
    "Agriculture",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sigmoid(value):
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def training_audit(rows):
    issues = {
        "missing_credit_score": 0,
        "missing_default_flag": 0,
        "negative_annual_revenue_usd": 0,
        "credit_score_out_of_range": 0,
        "debt_to_income_ratio_out_of_range": 0,
        "unknown_sector": 0,
        "missing_required_fields": 0,
    }
    clean_rows = []

    for row in rows:
        missing_required = any(str(row.get(field, "")).strip() == "" for field in TRAINING_REQUIRED_FIELDS)
        credit_score = safe_float(row.get("credit_score"))
        annual_revenue = safe_float(row.get("annual_revenue_usd"))
        dti = safe_float(row.get("debt_to_income_ratio"))
        default_flag = row.get("default_flag")
        sector = row.get("sector")

        invalid = False
        if missing_required:
            issues["missing_required_fields"] += 1
            invalid = True
        if credit_score is None:
            issues["missing_credit_score"] += 1
            invalid = True
        elif not 300 <= credit_score <= 850:
            issues["credit_score_out_of_range"] += 1
            invalid = True
        if default_flag is None or str(default_flag).strip() == "":
            issues["missing_default_flag"] += 1
            invalid = True
        if annual_revenue is None or annual_revenue < 0:
            if annual_revenue is not None and annual_revenue < 0:
                issues["negative_annual_revenue_usd"] += 1
            invalid = True
        if dti is None or not 0.05 <= dti <= 1.0:
            issues["debt_to_income_ratio_out_of_range"] += 1
            invalid = True
        if sector not in ALLOWED_SECTORS:
            issues["unknown_sector"] += 1
            invalid = True

        if not invalid:
            clean_rows.append(row)

    return issues, clean_rows


def scoring_audit(rows, model):
    issues = {
        "missing_credit_score": 0,
        "negative_annual_revenue_usd": 0,
        "credit_score_out_of_range": 0,
        "debt_to_income_ratio_out_of_range": 0,
        "missing_required_fields": 0,
        "unknown_category_values": 0,
    }
    clean_rows = []

    for row in rows:
        missing_required = any(str(row.get(field, "")).strip() == "" for field in SCORING_REQUIRED_FIELDS)
        credit_score = safe_float(row.get("credit_score"))
        annual_revenue = safe_float(row.get("annual_revenue_usd"))
        dti = safe_float(row.get("debt_to_income_ratio"))
        invalid = False

        if missing_required:
            issues["missing_required_fields"] += 1
            invalid = True
        if credit_score is None:
            issues["missing_credit_score"] += 1
            invalid = True
        elif not 300 <= credit_score <= 850:
            issues["credit_score_out_of_range"] += 1
            invalid = True
        if annual_revenue is None or annual_revenue < 0:
            if annual_revenue is not None and annual_revenue < 0:
                issues["negative_annual_revenue_usd"] += 1
            invalid = True
        if dti is None or not 0.05 <= dti <= 1.0:
            issues["debt_to_income_ratio_out_of_range"] += 1
            invalid = True

        for field in CATEGORICAL_FIELDS:
            allowed = set(model["categories"][field])
            if row.get(field) not in allowed:
                issues["unknown_category_values"] += 1
                invalid = True
                break

        if not invalid:
            clean_rows.append(row)

    return issues, clean_rows


def compute_stats(rows):
    stats = {}
    for field in NUMERIC_FIELDS:
        values = [safe_float(row[field]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std_dev = math.sqrt(variance) or 1.0
        stats[field] = {"mean": mean, "std": std_dev}
    return stats


def categories_from_rows(rows):
    return {
        field: sorted({row[field] for row in rows})
        for field in CATEGORICAL_FIELDS
    }


def encode_row(row, stats, categories):
    vector = [1.0]
    feature_names = ["bias"]
    contribution_details = []

    for field in NUMERIC_FIELDS:
        raw_value = safe_float(row[field])
        scaled_value = (raw_value - stats[field]["mean"]) / stats[field]["std"]
        vector.append(scaled_value)
        feature_name = f"num::{field}"
        feature_names.append(feature_name)
        contribution_details.append(
            {
                "feature": feature_name,
                "raw_value": raw_value,
                "encoded_value": scaled_value,
                "field": field,
                "kind": "numeric",
            }
        )

    for field in CATEGORICAL_FIELDS:
        for category in categories[field]:
            encoded = 1.0 if row[field] == category else 0.0
            feature_name = f"cat::{field}::{category}"
            vector.append(encoded)
            feature_names.append(feature_name)
            if encoded:
                contribution_details.append(
                    {
                        "feature": feature_name,
                        "raw_value": category,
                        "encoded_value": encoded,
                        "field": field,
                        "kind": "categorical",
                    }
                )

    return vector, feature_names, contribution_details


def build_matrix(rows, stats, categories):
    matrix = []
    labels = []
    feature_names = None
    contribution_map = []
    for row in rows:
        vector, current_feature_names, contribution_details = encode_row(row, stats, categories)
        matrix.append(vector)
        labels.append(int(row["default_flag"]))
        feature_names = current_feature_names
        contribution_map.append(contribution_details)
    return matrix, labels, feature_names, contribution_map


def train_logistic_regression(features, labels, epochs=2500, learning_rate=0.05, regularization=0.001):
    weights = [0.0 for _ in range(len(features[0]))]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_weight = (len(labels) / (2 * positive_count)) if positive_count else 1.0
    negative_weight = (len(labels) / (2 * negative_count)) if negative_count else 1.0

    for _ in range(epochs):
        gradients = [0.0 for _ in range(len(weights))]
        for vector, label in zip(features, labels):
            score = sum(weight * value for weight, value in zip(weights, vector))
            prediction = sigmoid(score)
            sample_weight = positive_weight if label == 1 else negative_weight
            error = (prediction - label) * sample_weight
            for index, value in enumerate(vector):
                gradients[index] += error * value

        scale = 1.0 / len(features)
        for index in range(len(weights)):
            penalty = regularization * weights[index] if index != 0 else 0.0
            weights[index] -= learning_rate * ((gradients[index] * scale) + penalty)

    return weights


def predict_probability(weights, vector):
    score = sum(weight * value for weight, value in zip(weights, vector))
    return sigmoid(score)


def split_rows(rows, seed=42, test_ratio=0.2):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    split_index = int(len(ordered) * (1 - test_ratio))
    return ordered[:split_index], ordered[split_index:]


def classification_metrics(actual, probabilities, threshold=0.5):
    predicted = [1 if value >= threshold else 0 for value in probabilities]
    tp = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    tn = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
    accuracy = (tp + tn) / len(actual) if actual else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def rank_feature_importance(weights, feature_names):
    rows = []
    for feature_name, weight in zip(feature_names[1:], weights[1:]):
        rows.append(
            {
                "feature": feature_name,
                "weight": round(weight, 6),
                "absolute_weight": abs(weight),
            }
        )
    rows.sort(key=lambda row: row["absolute_weight"], reverse=True)
    return rows


def save_model(path, stats, categories, weights, feature_names, thresholds, metrics, clean_rows):
    payload = {
        "model_type": "logistic_regression",
        "version": "1.0",
        "stats": stats,
        "categories": categories,
        "weights": weights,
        "feature_names": feature_names,
        "thresholds": thresholds,
        "training_row_count": len(clean_rows),
        "default_rate": round(sum(int(row["default_flag"]) for row in clean_rows) / len(clean_rows), 4),
        "metrics": metrics,
        "feature_importance": rank_feature_importance(weights, feature_names)[:15],
    }
    write_json(path, payload)


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def decision_label(probability, thresholds):
    if probability >= thresholds["decline"]:
        return "Decline"
    if probability >= thresholds["review"]:
        return "Review"
    return "Approve"


def format_currency(value):
    return "${:,.0f}".format(value)


def contribution_reason(detail, contribution):
    field = detail["field"]
    if detail["kind"] == "numeric":
        raw_value = detail["raw_value"]
        if field == "credit_score":
            if raw_value < 600:
                return f"Low credit score ({int(raw_value)}) materially increases default risk."
            return f"Credit score ({int(raw_value)}) is below the stronger-performing borrower range."
        if field == "debt_to_income_ratio":
            return f"Debt-to-income ratio ({raw_value:.2f}) is high relative to the training portfolio."
        if field == "interest_rate_pct":
            return f"Interest rate ({raw_value:.2f}%) is elevated, which often coincides with weaker borrower quality."
        if field == "loan_amount_usd":
            return f"Requested loan amount ({format_currency(raw_value)}) is large for this portfolio."
        if field == "annual_revenue_usd":
            return f"Annual revenue ({format_currency(raw_value)}) is low relative to requested credit."
        if field == "business_age_years":
            return f"Business operating history ({int(raw_value)} years) is limited."
        if field == "num_employees":
            return f"Employee count ({int(raw_value)}) is smaller than stronger-performing cohorts."
        if field == "loan_term_months":
            return f"Loan term ({int(raw_value)} months) is longer than the lower-risk cohort."
    if detail["kind"] == "categorical":
        category = detail["raw_value"]
        if field == "sector":
            return f"Sector exposure ({category}) has historically produced higher defaults in this sample."
        if field == "loan_purpose":
            return f"Loan purpose ({category}) has performed worse than the portfolio average."
        if field == "state":
            return f"State cohort ({category}) has a weaker repayment pattern in the training data."
    return f"{field} contributes positively to risk."


def make_brief(row, probability, decision, drivers):
    lead = drivers[0] if drivers else "The model found a weaker-than-average repayment profile."
    second = drivers[1] if len(drivers) > 1 else "No secondary driver rose materially above the rest."
    action = {
        "Decline": "Escalate for senior underwriter review and verify bank statements before issuing a final decline.",
        "Review": "Route to analyst review with refreshed cash-flow documents and recent bank statements.",
        "Approve": "Proceed with standard underwriting checks and confirm no new adverse information is present.",
    }[decision]
    return (
        f"RISK SUMMARY: {decision} at {probability:.1%} default probability. {lead}\n\n"
        f"KEY SIGNALS:\n"
        f"- {lead}\n"
        f"- {second}\n\n"
        f"ANALYST ACTION: {action}"
    )


def score_rows(rows, model):
    stats = model["stats"]
    categories = model["categories"]
    weights = model["weights"]
    thresholds = model["thresholds"]
    feature_names = model["feature_names"]
    name_to_weight = dict(zip(feature_names, weights))
    scored_rows = []

    for row in rows:
        vector, _, contribution_details = encode_row(row, stats, categories)
        probability = predict_probability(weights, vector)
        decision = decision_label(probability, thresholds)

        driver_rows = []
        for detail in contribution_details:
            weight = name_to_weight.get(detail["feature"], 0.0)
            contribution = weight * detail["encoded_value"]
            if contribution > 0:
                driver_rows.append((contribution, contribution_reason(detail, contribution)))
        driver_rows.sort(key=lambda item: item[0], reverse=True)
        drivers = [reason for _, reason in driver_rows[:2]]

        scored = dict(row)
        scored["default_probability"] = round(probability, 4)
        scored["decision"] = decision
        scored["underwriter_brief"] = make_brief(row, probability, decision, drivers)
        scored_rows.append(scored)

    return scored_rows


def training_summary(audit_issues, clean_rows, metrics, total_rows):
    return {
        "status": "READY" if clean_rows else "BLOCKED",
        "raw_row_count": total_rows,
        "clean_training_row_count": len(clean_rows),
        "default_rate": round(sum(int(row["default_flag"]) for row in clean_rows) / len(clean_rows), 4) if clean_rows else 0.0,
        "audit_issues": audit_issues,
        "model_metrics": metrics,
    }


def scoring_summary(audit_issues, total_rows, scored_rows):
    counts = Counter(row["decision"] for row in scored_rows)
    at_risk = sum(
        safe_float(row["loan_amount_usd"]) for row in scored_rows if row["decision"] in {"Decline", "Review"}
    )
    return {
        "status": "READY" if scored_rows else "BLOCKED",
        "input_row_count": total_rows,
        "scored_row_count": len(scored_rows),
        "flagged_row_count": counts["Decline"] + counts["Review"],
        "approve_count": counts["Approve"],
        "review_count": counts["Review"],
        "decline_count": counts["Decline"],
        "total_at_risk_usd": round(at_risk, 2),
        "audit_issues": audit_issues,
    }


def train_command(args):
    raw_rows = read_csv(args.historical)
    audit_issues, clean_rows = training_audit(raw_rows)
    if not clean_rows:
        raise SystemExit("No usable training rows found after audit.")

    if args.clean_out:
        write_csv(args.clean_out, clean_rows, fieldnames=raw_rows[0].keys())

    train_rows, test_rows = split_rows(clean_rows)
    stats = compute_stats(train_rows)
    categories = categories_from_rows(train_rows)
    train_x, train_y, feature_names, _ = build_matrix(train_rows, stats, categories)
    test_x, test_y, _, _ = build_matrix(test_rows, stats, categories)
    weights = train_logistic_regression(train_x, train_y)
    test_probabilities = [predict_probability(weights, vector) for vector in test_x]
    metrics = classification_metrics(test_y, test_probabilities)
    thresholds = {"review": args.review_threshold, "decline": args.decline_threshold}

    save_model(args.model_out, stats, categories, weights, feature_names, thresholds, metrics, clean_rows)

    summary = training_summary(audit_issues, clean_rows, metrics, len(raw_rows))
    if args.audit_out:
        write_json(args.audit_out, summary)

    print(json.dumps(summary, indent=2))


def score_command(args):
    model = load_model(args.model_in)
    application_rows = read_csv(args.applications)
    audit_issues, clean_rows = scoring_audit(application_rows, model)
    issue_count = sum(audit_issues.values())
    scored_rows = score_rows(clean_rows, model) if issue_count == 0 else []
    flagged_rows = [row for row in scored_rows if row["decision"] in {"Decline", "Review"}]

    decision_fields = list(application_rows[0].keys()) + ["default_probability", "decision", "underwriter_brief"]
    flagged_fields = [
        "loan_id",
        "sector",
        "state",
        "loan_amount_usd",
        "credit_score",
        "debt_to_income_ratio",
        "default_probability",
        "decision",
        "underwriter_brief",
    ]

    write_csv(args.decisions_out, scored_rows, decision_fields)
    write_csv(args.flagged_out, flagged_rows, flagged_fields)

    summary = scoring_summary(audit_issues, len(application_rows), scored_rows)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2))


def bootstrap_command(args):
    train_args = argparse.Namespace(
        historical=args.historical,
        model_out=args.model_out,
        clean_out=args.clean_out,
        audit_out=args.training_audit_out,
        review_threshold=args.review_threshold,
        decline_threshold=args.decline_threshold,
    )
    train_command(train_args)

    score_args = argparse.Namespace(
        applications=args.applications,
        model_in=args.model_out,
        decisions_out=args.decisions_out,
        flagged_out=args.flagged_out,
        summary_out=args.scoring_summary_out,
    )
    score_command(score_args)


def build_parser():
    parser = argparse.ArgumentParser(description="FinanceFlow underwriting training and scoring pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Audit historical data and train the classifier")
    train_parser.add_argument("--historical", required=True)
    train_parser.add_argument("--model-out", required=True)
    train_parser.add_argument("--clean-out")
    train_parser.add_argument("--audit-out")
    train_parser.add_argument("--review-threshold", type=float, default=0.4)
    train_parser.add_argument("--decline-threshold", type=float, default=0.65)
    train_parser.set_defaults(func=train_command)

    score_parser = subparsers.add_parser("score", help="Score new applications with a saved model")
    score_parser.add_argument("--applications", required=True)
    score_parser.add_argument("--model-in", required=True)
    score_parser.add_argument("--decisions-out", required=True)
    score_parser.add_argument("--flagged-out", required=True)
    score_parser.add_argument("--summary-out", required=True)
    score_parser.set_defaults(func=score_command)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Train the model and score a sample file")
    bootstrap_parser.add_argument("--historical", required=True)
    bootstrap_parser.add_argument("--applications", required=True)
    bootstrap_parser.add_argument("--model-out", required=True)
    bootstrap_parser.add_argument("--clean-out", required=True)
    bootstrap_parser.add_argument("--training-audit-out", required=True)
    bootstrap_parser.add_argument("--decisions-out", required=True)
    bootstrap_parser.add_argument("--flagged-out", required=True)
    bootstrap_parser.add_argument("--scoring-summary-out", required=True)
    bootstrap_parser.add_argument("--review-threshold", type=float, default=0.4)
    bootstrap_parser.add_argument("--decline-threshold", type=float, default=0.65)
    bootstrap_parser.set_defaults(func=bootstrap_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
