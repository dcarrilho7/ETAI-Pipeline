"""Run cleaning, preprocessing, evaluation, and optional tuning."""

import shutil
import tempfile
from datetime import datetime

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.model import build_pipeline
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    df = add_engineered_features(df)

    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"]
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config.get("shuffle", True),
        random_state=cv_config.get("random_state", 42),
    )

    # A majority-class dummy gives a simple floor for the other models.
    dummy_config = {"type": "dummy", "params": {"strategy": "most_frequent"}}
    dummy = build_pipeline(config["preprocessing"], dummy_config)
    floor_scores, _ = cross_validate_pipeline(
        dummy, X_dev, y_dev, cv, {selection_metric: scorers[selection_metric]},
        selection_metric=selection_metric, n_jobs=cv_config.get("n_jobs", 1),
    )
    floor_mean = floor_scores["validation"].mean()
    print(f"Dummy baseline ({selection_metric}): {floor_mean:.3f}")

    selection_config = config.get("feature_selection") or {}
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")
    estimator = build_pipeline(
        config["preprocessing"],
        config["model"],
        selection_config,
        memory=cache_dir,
    )
    fold_scores, oof_proba = cross_validate_pipeline(
        estimator, X_dev, y_dev, cv, scorers,
        selection_metric=selection_metric, n_jobs=cv_config.get("n_jobs", 1),
    )
    report = cv_report(fold_scores, metrics, selection_metric)

    tuning_config = config.get("tuning", {})
    tuned_mean = None
    tuned_std = None
    study = None
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_space = tuning_config.get("search_spaces", {}).get(model_type)
        if search_space is None:
            raise ValueError(
                f"Tuning is enabled but config.yaml has no search space for '{model_type}'."
            )
        search_space = dict(search_space)
        if selection_config.get("enabled", False):
            search_space.update(selection_config.get("search_space", {}))

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
        nested_scores, oof_proba = nested_cross_validate(
            estimator, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=cv_config.get("n_jobs", 1),
        )
        tuned_values = nested_scores[f"validation_{selection_metric}"]
        tuned_mean = tuned_values.mean()
        tuned_std = tuned_values.std(ddof=1)
        report += "\n\nNested cross-validation:\n"
        report += cv_report(nested_scores, metrics, selection_metric)

        estimator, study = tune_pipeline(
            estimator, X_dev, y_dev, inner_cv, scorers[selection_metric],
            search_space, n_trials, tuning_seed, cv_config.get("n_jobs", 1),
        )
        report += "\n\n" + tuning_report(
            study, nested_scores, selection_metric
        )

    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(
        y_dev, oof_proba, extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    estimator.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if study is not None:
        refit += f" Tuned hyperparameters: {study.best_params}"
    if selection_config.get("enabled", False):
        all_features = estimator.named_steps["prep"].get_feature_names_out()
        kept_features = estimator.named_steps["select"].get_feature_names_out(all_features)
        refit += f"\nFeatures kept: {len(kept_features)} of {len(all_features)}: {', '.join(kept_features)}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (
        f"Locked test set: {len(X_test)} rows set aside and not evaluated. "
        f"Development set: {len(X_dev)} rows."
    )
    print(locked)
    report += "\n" + locked + "\n"

    scoreboard_path = config.get("output", {}).get(
        "scoreboard", "results/runs.csv"
    )
    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": config["model"]["type"],
        "tuned": bool(study is not None),
        "encoder": config["preprocessing"].get("encoder"),
        "scaler": config["preprocessing"].get("scaler"),
        "feature_selection": (
            f"min_votes {selection_config.get('min_votes', 2)}"
            if selection_config.get("enabled", False) else "off"
        ),
        "metric": selection_metric,
        "untuned_mean": fold_scores[f"validation_{selection_metric}"].mean(),
        "untuned_std": fold_scores[f"validation_{selection_metric}"].std(ddof=1),
        "tuned_mean": tuned_mean,
        "tuned_std": tuned_std,
        "floor_mean": floor_mean,
        "best_trial": study.best_value if study is not None else None,
        "tuned_params": repr(study.best_params) if study is not None else "",
        **{
            f"untuned_fold_{i}": value
            for i, value in enumerate(
                fold_scores[f"validation_{selection_metric}"], start=1
            )
        },
        **({
            f"tuned_fold_{i}": value
            for i, value in enumerate(
                nested_scores[f"validation_{selection_metric}"], start=1
            )
        } if tuned_mean is not None else {}),
    }
    save_scores(scoreboard_path, row)
    print(f"Scoreboard saved to {scoreboard_path}")

    report += "\n\nScoreboard row:\n" + str(row) + "\n"
    shutil.rmtree(cache_dir, ignore_errors=True)
    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
