"""Run cleaning, leak-safe preprocessing, evaluation, and optional tuning."""

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.model import build_pipeline
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))

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

    estimator = build_pipeline(config["preprocessing"], config["model"])

    cv_config = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config.get("shuffle", True),
        random_state=cv_config.get("random_state", 42),
    )
    fold_scores, y_oof = cross_validate_pipeline(
        estimator,
        X_dev,
        y_dev,
        cv=cv,
        scoring=cv_config.get("scoring", "accuracy"),
        n_jobs=cv_config.get("n_jobs", 1),
    )
    report = cv_report(fold_scores, cv_config.get("scoring", "accuracy"))

    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(
                f"Tuning is enabled but config.yaml has no search space for '{model_type}'."
            )

        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        print(
            f"\nNested cross-validation: {cv.get_n_splits()} outer folds x "
            f"{n_trials} trials x {inner_cv.get_n_splits()} inner folds ..."
        )
        nested_scores, y_oof = nested_cross_validate(
            estimator,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            cv_config.get("scoring", "accuracy"),
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=cv_config.get("n_jobs", 1),
        )
        scoring = cv_config.get("scoring", "accuracy")
        report += "\n\nNested cross-validation (honest tuning estimate):\n"
        report += cv_report(nested_scores, scoring)

        estimator, study = tune_pipeline(
            estimator,
            X_dev,
            y_dev,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=cv_config.get("n_jobs", 1),
        )
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    estimator.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_config.get("enabled", False):
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (
        f"Locked test set: {len(X_test)} rows set aside and not evaluated. "
        f"Development set: {len(X_dev)} rows."
    )
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
