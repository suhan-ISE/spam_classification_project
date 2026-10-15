from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="CSV file path or folder containing the dataset CSV.")
    parser.add_argument("--text-column", default=None, help="Explicit email-text column name.")
    parser.add_argument("--label-column", default=None, help="Explicit target-label column name.")


def find_csv(data_path: Path) -> Path:
    data_path = data_path.expanduser().resolve()
    if data_path.is_file() and data_path.suffix.lower() == ".csv":
        return data_path
    if data_path.is_dir():
        csvs = sorted(data_path.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        if not csvs:
            raise FileNotFoundError(
                f"No CSV found in {data_path}. Download the required Kaggle dataset and put its CSV there."
            )
        raise ValueError(
            f"Multiple CSV files found in {data_path}: {[p.name for p in csvs]}. "
            "Pass the exact CSV path using --data."
        )
    raise FileNotFoundError(f"Data path does not exist or is not a CSV: {data_path}")


def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_dataset(data_path: Path, text_column: str | None = None,
                 label_column: str | None = None) -> tuple[pd.DataFrame, str, str]:
    csv_path = find_csv(data_path)
    df = pd.read_csv(csv_path, encoding_errors="replace")
    if df.empty:
        raise ValueError(f"The CSV is empty: {csv_path}")

    normalized = {_norm_col(c): c for c in df.columns}
    text_candidates = [
        "emailtext", "text", "email", "message", "body", "content",
        "mailtext", "emailbody", "v2"
    ]
    label_candidates = [
        "emailtype", "label", "category", "target", "class", "spamham",
        "type", "v1"
    ]

    if text_column is None:
        for candidate in text_candidates:
            if candidate in normalized and normalized[candidate] != label_column:
                text_column = normalized[candidate]
                break
    if label_column is None:
        for candidate in label_candidates:
            if candidate in normalized and normalized[candidate] != text_column:
                label_column = normalized[candidate]
                break

    if text_column not in df.columns if text_column is not None else True:
        raise ValueError(
            f"Could not detect the email-text column. Columns found: {list(df.columns)}. "
            "Pass --text-column with the correct name."
        )
    if label_column not in df.columns if label_column is not None else True:
        raise ValueError(
            f"Could not detect the label column. Columns found: {list(df.columns)}. "
            "Pass --label-column with the correct name."
        )
    if text_column == label_column:
        raise ValueError("Text and label columns must be different.")

    data = df[[text_column, label_column]].copy()
    data.columns = ["text", "raw_label"]
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    
    data["raw_label"] = data["raw_label"].astype("string").str.strip()

    data = data.dropna(subset=["raw_label"])
    data = data[data["text"].ne("")]
    data = data[data["raw_label"].ne("")]

    # Keep only recognized spam/ham labels.
    # This removes accidental metadata rows such as {"mode":"full".
    valid_labels = {
        "ham", "spam",
        "not spam", "not_spam",
        "legitimate", "legit",
        "0", "1", "true", "false", "yes", "no"
    }

    data["normalized_label"] = data["raw_label"].str.lower().str.strip()
    invalid_rows = ~data["normalized_label"].isin(valid_labels)

    if invalid_rows.any():
        print(f"Removing {int(invalid_rows.sum())} rows with invalid labels:")
        print(data.loc[invalid_rows, "raw_label"].value_counts().to_dict())

    data = data.loc[~invalid_rows].copy()
    data = data.drop_duplicates(
        subset=["text", "raw_label"]
    ).reset_index(drop=True)


    if data["raw_label"].nunique() != 2:
        raise ValueError(
            f"Expected exactly two classes (spam and ham), found: "
            f"{data['raw_label'].value_counts().to_dict()}"
        )

    # Map semantic labels to 1=spam and 0=ham where possible.
    label_norm = data["raw_label"].str.lower().str.strip()
    unique_norm = list(label_norm.unique())
    spam_terms = {"spam", "junk", "1", "true", "yes"}
    ham_terms = {"ham", "not spam", "not_spam", "legitimate", "legit", "0", "false", "no"}
    if set(unique_norm).issubset(spam_terms | ham_terms) and any(x in spam_terms for x in unique_norm):
        mapping = {x: (1 if x in spam_terms else 0) for x in unique_norm}
        data["label"] = label_norm.map(mapping).astype(int)
        label_mapping = mapping
    else:
        # Stable deterministic encoding; inspect the printed mapping in the run output.
        sorted_labels = sorted(unique_norm)
        mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1}
        data["label"] = label_norm.map(mapping).astype(int)
        label_mapping = mapping

    data = data[["text", "label"]].reset_index(drop=True)
    data.attrs["csv_path"] = str(csv_path)
    data.attrs["text_column"] = text_column
    data.attrs["label_column"] = label_column
    data.attrs["label_mapping"] = label_mapping
    return data, text_column, label_column


def split_data(data: pd.DataFrame):
    """Stratified 70/15/15 split, with deterministic seed."""
    train_df, temp_df = train_test_split(
        data, test_size=0.30, random_state=RANDOM_STATE, stratify=data["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Ham (0)", "Spam (1)"], zero_division=0, output_dict=True
        ),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_dataset_summary(data: pd.DataFrame, csv_path: str, text_column: str,
                          label_column: str, label_mapping: dict) -> None:
    print(f"Dataset: {csv_path}")
    print(f"Text column: {text_column!r}; label column: {label_column!r}")
    print(f"Rows after dropping missing/empty and exact duplicate text-label pairs: {len(data)}")
    print(f"Label mapping (normalized source label -> 0/1): {label_mapping}")
    print("Label counts (0=ham if semantic labels were recognized; 1=spam):")
    print(data["label"].map({0: "Ham (0)", 1: "Spam (1)"}).value_counts().to_string())
