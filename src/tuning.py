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
    scorer,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Tune a pipeline using Optuna and one selected CV scorer."""

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            parameter_type = spec["type"]
            if parameter_type == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif parameter_type == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif parameter_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type for {name}: {parameter_type}")

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    return clone(pipeline).set_params(**study.best_params), study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scorers,
    selection_metric: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Estimate tuning on unseen outer folds and return OOF probabilities."""
    selection_scorer = scorers[selection_metric]
    rows = []
    oof_proba = np.empty(len(X), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(
            pipeline, X_train, y_train, inner_cv, selection_scorer,
            search_space, n_trials, random_state, n_jobs,
        )
        best_pipeline.fit(X_train, y_train)

        row = {"fold": fold, "inner_best": study.best_value}
        for metric, scorer in scorers.items():
            train_score = scorer(best_pipeline, X_train, y_train)
            validation_score = scorer(best_pipeline, X_val, y_val)
            row[f"train_{metric}"] = train_score
            row[f"validation_{metric}"] = validation_score
            row[f"gap_{metric}"] = train_score - validation_score
        row["train"] = row[f"train_{selection_metric}"]
        row["validation"] = row[f"validation_{selection_metric}"]
        row["gap"] = row[f"gap_{selection_metric}"]
        row.update({f"best_{key}": value for key, value in study.best_params.items()})
        rows.append(row)
        oof_proba[val_idx] = best_pipeline.predict_proba(X_val)[:, 1]

    return pd.DataFrame(rows), oof_proba


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    selection_metric: str = "accuracy",
    top: int = 5,
) -> str:
    """Report the selected parameters and the nested-CV estimate."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(
        columns=lambda column: column.replace("params_", "").replace("user_attrs_", "")
    )
    trials = trials.rename(
        columns={"number": "trial", "value": f"mean {selection_metric}"}
    )

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores[f"validation_{selection_metric}"].mean()
    lines = [
        f"Tuning on the development set ({len(study.trials)} Optuna trials, metric: {selection_metric})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (best trial score)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {selection_metric}", ascending=False)
        .head(top)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Nested CV estimate:",
        f"  inner tuning score mean = {inner:.3f}",
        f"  outer validation score mean = {outer:.3f}",
        f"  difference = {inner - outer:+.3f}",
    ]
    text = "\n".join(lines)
    print(text)
    return text
