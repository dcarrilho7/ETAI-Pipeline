"""Run cleaning, leak-safe preprocessing, model fitting, and evaluation."""

import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.evaluate import cross_validate_pipeline, cv_report, evaluate, fairness_report
from src.model import build_model
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    split_dev_test,
    split_features_target,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    df = clean_dataset(
        load_data(config["data"]["path"]),
        config["diagnostics"],
    )

    X, y, extras = split_features_target(
        df,
        config["data"],
        config["preprocessing"],
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    estimator = Pipeline([
        ("preprocessor", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config.get("shuffle", True),
        random_state=cv_config.get("random_state", 42),
    )
    fold_scores = cross_validate_pipeline(
        estimator,
        X_dev,
        y_dev,
        cv=cv,
        scoring=cv_config.get("scoring", "accuracy"),
        n_jobs=cv_config.get("n_jobs", 1),
    )
    report = cv_report(fold_scores, cv_config.get("scoring", "accuracy"))

    # Refit the selected pipeline on all development rows, then assess the locked test set.
    estimator.fit(X_dev, y_dev)
    y_dev_pred = estimator.predict(X_dev)
    y_test_pred = estimator.predict(X_test)
    report += "\n\nFinal locked-test evaluation:\n"
    report += evaluate(y_dev, y_dev_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test,
        y_test_pred,
        extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
