"""Data loading."""
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)
