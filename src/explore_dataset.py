import argparse
from pathlib import Path
import pandas as pd

from common import add_data_arguments, load_dataset, print_dataset_summary, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Explore the Kaggle spam-email dataset.")
    add_data_arguments(parser)
    args = parser.parse_args()

    data, text_col, label_col = load_dataset(args.data, args.text_column, args.label_column)
    print_dataset_summary(
        data, data.attrs["csv_path"], text_col, label_col, data.attrs["label_mapping"]
    )

    raw = pd.read_csv(data.attrs["csv_path"], encoding_errors="replace")
    lengths = data["text"].str.split().str.len()
    profile = [
        "DATASET PROFILE",
        f"Source CSV: {data.attrs['csv_path']}",
        f"Original columns: {list(raw.columns)}",
        f"Detected text column: {text_col}",
        f"Detected label column: {label_col}",
        f"Original row count: {len(raw)}",
        f"Rows after cleaning: {len(data)}",
        f"Rows removed during cleaning: {len(raw) - len(data)}",
        f"Missing values by original column:\n{raw.isna().sum().to_string()}",
        f"Exact duplicate rows in original data: {int(raw.duplicated().sum())}",
        f"Class distribution:\n{data['label'].map({0: 'Ham (0)', 1: 'Spam (1)'}).value_counts().to_string()}",
        f"Word-count summary:\n{lengths.describe().to_string()}",
        "",
        "Cleaning decisions: rows with missing labels, empty text, or exact duplicate text-label pairs are removed.",
        "Inspect the detected label mapping before interpreting spam-specific metrics.",
        "The training scripts use a stratified 70/15/15 train/validation/test split.",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "dataset_profile.txt"
    out.write_text("\n\n".join(profile), encoding="utf-8")
    print(f"\nSaved profile to: {out}")


if __name__ == "__main__":
    main()
