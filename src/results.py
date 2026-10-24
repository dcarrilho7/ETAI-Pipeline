"""Save the text report and the short scoreboard for each run."""

import os
from datetime import datetime

import pandas as pd


def save_scores(path: str, row: dict) -> str:
    """Append one run to the CSV scoreboard."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_row = pd.DataFrame([row])
    if os.path.exists(path):
        old_rows = pd.read_csv(path)
        new_row = pd.concat([old_rows, new_row], ignore_index=True)
    new_row.to_csv(path, index=False)
    return path


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """Write the full report for one run to a timestamped text file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")
    test_config = config.get("test_set", {})
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params', {})}\n"
        f"Locked test size: {test_config.get('size')}  "
        f"random_state: {test_config.get('random_state')}\n"
        f"CV: {config.get('cv', {})}\n"
        f"Preprocessing: {config.get('preprocessing', {}).get('imputation', {})}\n"
        f"Tuning: {config.get('tuning', {})}\n"
        + "=" * 60 + "\n\n"
    )
    with open(path, "w") as file:
        file.write(header + report_text)
    return path
