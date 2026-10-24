"""Evaluation helpers used by the cross-validation part of the pipeline."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics):
    """Build sklearn scorers from the names in config.yaml."""
    scorers = {}
    for metric in metrics:
        if metric == "fbeta_2":
            scorers[metric] = make_scorer(fbeta_score, beta=2, zero_division=0)
        else:
            scorers[metric] = get_scorer(metric)
    return scorers


def cross_validate_pipeline(
    pipeline, X, y, cv, scorers, selection_metric="accuracy", n_jobs=1
):
    """Run CV for all configured metrics and return out-of-fold probabilities."""
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    rows = []
    oof_proba = np.empty(len(X), dtype=float)
    for fold in range(len(scores[f"test_{selection_metric}"])):
        row = {"fold": fold + 1}
        for metric in scorers:
            train = scores[f"train_{metric}"][fold]
            validation = scores[f"test_{metric}"][fold]
            row[f"train_{metric}"] = train
            row[f"validation_{metric}"] = validation
            row[f"gap_{metric}"] = train - validation
        row["train"] = row[f"train_{selection_metric}"]
        row["validation"] = row[f"validation_{selection_metric}"]
        row["gap"] = row[f"gap_{selection_metric}"]
        rows.append(row)

    for estimator, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = estimator.predict_proba(X.iloc[val_idx])[:, 1]

    return pd.DataFrame(rows), oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics, selection_metric="accuracy") -> str:
    """Format fold-level means and standard deviations for the report."""
    lines = [
        f"Cross-validation (selection metric: {selection_metric}):",
        fold_scores.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
        "",
    ]
    for metric in metrics:
        values = fold_scores[f"validation_{metric}"]
        gaps = fold_scores[f"gap_{metric}"]
        lines.append(
            f"{metric}: {values.mean():.3f} +/- {values.std(ddof=1):.3f} "
            f"(mean gap {gaps.mean():+.3f})"
        )
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, oof_proba, threshold=0.5) -> str:
    """Create a classification report from out-of-fold probabilities."""
    y_pred = (np.asarray(oof_proba) >= threshold).astype(int)
    lines = [
        "Classification report (out-of-fold predictions, development set):",
        f"Confusion matrix (threshold={threshold}):",
        str(confusion_matrix(y_true, y_pred)),
        classification_report(y_true, y_pred, zero_division=0),
    ]
    text = "\n".join(lines)
    print(text)
    return text


def fairness_report(
    y_true, oof_proba, extras: pd.DataFrame, threshold=0.5, sensitive_attr="race"
) -> str:
    """Compare model and COMPAS false-positive rates by race."""
    df = extras.reset_index(drop=True).copy()
    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = (np.asarray(oof_proba) >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, group_df in df.groupby(sensitive_attr):
            negatives = group_df[group_df["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


def corrected_paired_test(scores_a, scores_b, test_train_ratio):
    """Return the corrected resampled t-test used for two CV score vectors."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    differences = scores_a - scores_b
    variance = np.var(differences, ddof=1)
    correction = (1 / len(differences)) + test_train_ratio
    t_stat = differences.mean() / np.sqrt(correction * variance)
    p_value = 2 * stats.t.sf(abs(t_stat), df=len(differences) - 1)
    return float(t_stat), float(p_value)


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """Keep the original simple holdout report available for older runs."""
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    text = "\n".join([
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {train_accuracy - test_accuracy:+.3f}",
        "",
        "Classification report (test set):",
        classification_report(y_test, y_pred, zero_division=0),
    ])
    print(text)
    return text
