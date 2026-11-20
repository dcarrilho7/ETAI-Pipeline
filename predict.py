"""Predict new rows with a fitted pipeline saved by main.py."""

import argparse

import joblib
import pandas as pd

from src.data import load_config, load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    clean = clean_dataset(df, config["diagnostics"])
    clean = add_engineered_features(clean)
    X, _ = split_features(clean, config["data"], config["preprocessing"])
    probability = pipeline.predict_proba(X)[:, 1]
    threshold = (config.get("prediction") or {}).get("threshold", 0.5)
    id_column = config["data"].get("id_column", config["diagnostics"].get("id_column", "id"))
    return pd.DataFrame({
        id_column: df[id_column].to_numpy(),
        "probability": probability,
        "prediction": (probability >= threshold).astype(int),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with the same features as the training data")
    parser.add_argument("output", nargs="?", default="predictions.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(load_data(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")
