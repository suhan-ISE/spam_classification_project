import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from common import (
    add_data_arguments, load_dataset, split_data, calculate_metrics, save_json,
    print_dataset_summary, OUTPUT_DIR
)


def plot_confusions(results, out_path: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, results.items()):
        sns.heatmap(
            result["metrics"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham (0)", "Spam (1)"], yticklabels=["Ham (0)", "Spam (1)"], ax=ax
        )
        ax.set_title(f"{name}\nTest confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train traditional TF-IDF spam classifiers.")
    add_data_arguments(parser)
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    data, text_col, label_col = load_dataset(args.data, args.text_column, args.label_column)
    print_dataset_summary(data, data.attrs["csv_path"], text_col, label_col, data.attrs["label_mapping"])
    train_df, val_df, test_df = split_data(data)
    print(f"Split sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    # TF-IDF learns vocabulary and inverse-document frequencies from training text only.
    # Keep tokenization simple and reproducible; the vectorizer lowercases by default.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=None, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    results = {}
    for name, classifier in models.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=1,
                sublinear_tf=True
            )),
            ("classifier", classifier),
        ])
        start = time.perf_counter()
        model.fit(train_df["text"], train_df["label"])
        train_seconds = time.perf_counter() - start

        # Validation metrics are recorded for monitoring; final reported comparison uses test set.
        val_pred = model.predict(val_df["text"])
        test_pred = model.predict(test_df["text"])
        metrics = calculate_metrics(test_df["label"], test_pred)
        results[name] = {
            "model": name,
            "metrics": metrics,
            "validation_f1": float(__import__("sklearn.metrics").metrics.f1_score(
                val_df["label"], val_pred, zero_division=0
            )),
            "training_time_seconds": train_seconds,
            "model_type": "traditional",
            "test_rows": int(len(test_df)),
        }
        print(f"\n{name}")
        print(f"Training time: {train_seconds:.2f} seconds")
        print(f"Test metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
              f"recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json({
        "source_csv": data.attrs["csv_path"],
        "text_column": text_col,
        "label_column": label_col,
        "label_mapping": data.attrs["label_mapping"],
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df), "seed": 42},
        "models": results,
    }, OUTPUT_DIR / "traditional_metrics.json")
    rows = []
    for name, item in results.items():
        m = item["metrics"]
        rows.append({
            "model": name, "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1_score": m["f1_score"],
            "training_time_seconds": item["training_time_seconds"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "traditional_metrics.csv", index=False)
    plot_confusions(results, OUTPUT_DIR / "traditional_confusion_matrices.png")
    print(f"\nSaved metrics and plot under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
