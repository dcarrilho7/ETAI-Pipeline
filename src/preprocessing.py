"""Cleaning and leak-safe preprocessing for the modelling pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)



def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Convert values that violate configured domain rules into missing values."""
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = (
            numeric >= bounds["min"]
            if "min" in bounds
            else pd.Series(True, index=numeric.index)
        )
        upper_ok = (
            numeric <= bounds["max"]
            if "max" in bounds
            else pd.Series(True, index=numeric.index)
        )
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({
            "column": column,
            "rule": bounds,
            "violations": int(violations.sum()),
        })
        df.loc[violations, column] = np.nan

    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for column, mapping in columns_and_maps.items():
        if column not in out.columns:
            continue
        cleaned = out[column].astype("string").str.strip()
        out[column] = cleaned.str.lower().map(mapping).fillna(cleaned)
        out.loc[out[column].astype("string").str.strip().isin(placeholder_tokens), column] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Apply deterministic cleaning rules before any fitted preprocessing."""
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for column in diagnostics_config.get("numeric_text_columns", []):
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column].replace(list(placeholder_tokens), np.nan),
                errors="coerce",
            )

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = _canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens,
    )

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """Remove duplicate training rows before the development/test split."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}_was_missing"] = out[column].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, preprocessing_config: dict):
    """Separate features, target, and fairness-audit columns."""
    df = add_missingness_indicators(
        df, preprocessing_config.get("mnar_indicator_sources", [])
    )
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    y = df[target]

    extras_columns = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_columns].copy()
    columns_to_exclude = set(data_config.get("drop_columns", [])) | {target, sensitive_attr}
    X = df[[c for c in df.columns if c not in columns_to_exclude]]
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Create the locked test set; model selection happens only on the development set."""
    return train_test_split(
        X, y, extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Build preprocessing whose fitted values are learned inside each CV fold."""
    numeric = preprocessing_config["numeric_features"]
    categorical = preprocessing_config["categorical_features"]
    imputation = preprocessing_config.get("imputation", {})
    scaler_name = preprocessing_config.get("scaler", "robust")
    encoder_name = preprocessing_config.get("encoder", "onehot")
    scaler = {
        "none": "passthrough",
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }[scaler_name]

    if imputation.get("numeric_strategy", "median") == "knn":
        numeric_imputer = KNNImputer(n_neighbors=imputation.get("n_neighbors", 5))
    else:
        numeric_imputer = SimpleImputer(
            strategy=imputation.get("numeric_strategy", "median")
        )

    numeric_pipeline = Pipeline([
        ("impute", numeric_imputer),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy=imputation.get("categorical_strategy", "most_frequent")
        )),
        ("encode", _build_encoder(encoder_name, preprocessing_config.get("random_state", 42))),
    ])

    indicator_columns = [
        f"{column}_was_missing"
        for column in preprocessing_config.get("mnar_indicator_sources", [])
    ]
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
        ("indicators", "passthrough", indicator_columns),
    ])


def _build_encoder(name: str, random_state: int):
    """Create the categorical encoder selected in config.yaml."""
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "target":
        return TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
        )
    raise ValueError("Unknown encoder: {}. Choose 'onehot' or 'target'.".format(name))
