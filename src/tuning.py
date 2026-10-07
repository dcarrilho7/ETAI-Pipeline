"""Optuna hyperparameter tuning and nested cross-validation."""

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


def tune_pipeline(
    pipeline,
    X,
    y,
    cv,
    scoring: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Tune a pipeline using Optuna and CV on the development data."""

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            parameter_type = spec["type"]
            if parameter_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif parameter_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif parameter_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(
                    f"Unknown search-space type for {name}: {parameter_type}. "
                    "Choose int, float, or categorical."
                )

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scoring: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Estimate the full tune-then-refit procedure with nested CV."""
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            random_state,
            n_jobs,
        )
        best_pipeline.fit(X_train, y_train)

        train_score = scorer(best_pipeline, X_train, y_train)
        validation_score = scorer(best_pipeline, X_val, y_val)
        y_oof[val_idx] = best_pipeline.predict(X_val)
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": validation_score,
            "gap": train_score - validation_score,
            "inner_best": study.best_value,
            **study.best_params,
        })

    return pd.DataFrame(rows), y_oof


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    scoring: str = "accuracy",
    top: int = 5,
) -> str:
    """Report selected parameters and the nested-CV optimism check."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda column: column.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": f"mean {scoring}"})

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()
    lines = [
        f"Tuning on the development set ({len(study.trials)} Optuna trials, metric: {scoring})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (optimistic best-trial score)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {scoring}", ascending=False)
        .head(top)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  inner tuning score mean = {inner:.3f}",
        f"  outer validation score mean = {outer:.3f}",
        f"  optimism = {inner - outer:+.3f}   -> report the outer score",
    ]
    text = "\n".join(lines)
    print(text)
    return text
