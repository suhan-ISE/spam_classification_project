import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import OUTPUT_DIR


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    traditional_path = OUTPUT_DIR / "traditional_metrics.json"
    if traditional_path.exists():
        data = json.loads(traditional_path.read_text(encoding="utf-8"))
        for name, item in data.get("models", {}).items():
            m = item["metrics"]
            rows.append({
                "model": name, "accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1_score": m["f1_score"],
                "training_time_seconds": item["training_time_seconds"],
            })

    transformer_path = OUTPUT_DIR / "distilbert_metrics.json"
    if transformer_path.exists():
        data = json.loads(transformer_path.read_text(encoding="utf-8"))
        m = data["metrics"]
        rows.append({
            "model": "DistilBERT", "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1_score": m["f1_score"],
            "training_time_seconds": data["training_time_seconds"],
        })

    if not rows:
        raise FileNotFoundError(
            "No results found. Run train_traditional.py and/or train_distilbert.py first."
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "all_model_comparison.csv", index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    melted = results.melt(id_vars="model", value_vars=metric_cols,
                          var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model")
    plt.ylim(0, 1.05)
    plt.title("Spam classification — test metric comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="training_time_seconds")
    plt.title("Model training-time comparison")
    plt.xlabel("Model")
    plt.ylabel("Training time (seconds; hardware-dependent)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_time_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(results.to_string(index=False))
    print(f"\nSaved comparison CSV and plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
