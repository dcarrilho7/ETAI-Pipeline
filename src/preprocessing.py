"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_diagnostics import flag_invalid_values


def _canonicalize_categories(
    df: pd.DataFrame,
    columns_and_maps: dict,
    placeholder_tokens: set
) -> pd.DataFrame:

    out = df.copy()

    for col, mapping in columns_and_maps.items():

        if col not in out.columns:
            continue

        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()

        out[col] = lowered.map(mapping).fillna(cleaned)

        out.loc[
            out[col].astype(str).str.strip().isin(placeholder_tokens),
            col
        ] = np.nan

    return out

def clean_dataset(
    df: pd.DataFrame,
    diagnostics_config: dict
) -> pd.DataFrame:

    out = df.copy()

    placeholder_tokens = set(
        diagnostics_config.get("placeholder_tokens", [])
    )

    # Columns that should be numeric but may contain text placeholders
    for col in diagnostics_config.get("numeric_text_columns", []):

        if col in out.columns:

            out[col] = pd.to_numeric(
                out[col].replace(
                    list(placeholder_tokens),
                    np.nan
                ),
                errors="coerce"
            )

    # Invalid values become NaN
    flag_invalid_values(
        out,
        diagnostics_config.get("validity_rules", {})
    )

    # Clean categorical values
    out = _canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens
    )

    # Remove duplicated rows
    out = out.drop_duplicates()

    # Remove repeated IDs
    id_column = diagnostics_config.get("id_column")

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(
            subset=id_column,
            keep="first"
        )

    return out


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
