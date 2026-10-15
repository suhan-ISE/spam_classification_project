import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    add_data_arguments, load_dataset, split_data, calculate_metrics, save_json,
    print_dataset_summary, OUTPUT_DIR
)

MODEL_NAME = "distilbert-base-uncased"


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            batches += 1
            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
    return (total_loss / max(batches, 1), all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT using a custom PyTorch loop.")
    add_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze DistilBERT encoder; only train classification head.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data, text_col, label_col = load_dataset(args.data, args.text_column, args.label_column)
    print_dataset_summary(data, data.attrs["csv_path"], text_col, label_col, data.attrs["label_mapping"])
    train_df, val_df, test_df = split_data(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: CPU training may be very slow. Consider --epochs 1 --batch-size 4 --max-length 128 for a smoke test.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = EmailDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    val_ds = EmailDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)
    test_ds = EmailDataset(test_df["text"], test_df["label"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    if args.freeze_base:
        for param in model.distilbert.parameters():
            param.requires_grad = False
    model.to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, weight_decay=0.01
    )

    history = {"epoch": [], "train_loss": [], "validation_loss": [], "validation_f1": []}
    start_all = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss, batches = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(batches, 1)
        val_loss, val_labels, val_preds = evaluate(model, val_loader, device)
        val_metrics = calculate_metrics(val_labels, val_preds)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_f1"].append(val_metrics["f1_score"])
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
              f"validation_loss={val_loss:.4f}, validation_F1={val_metrics['f1_score']:.4f}")

    training_seconds = time.perf_counter() - start_all
    test_loss, test_labels, test_preds = evaluate(model, test_loader, device)
    metrics = calculate_metrics(test_labels, test_preds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "distilbert_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "distilbert_training_history.csv", index=False)

    payload = {
        "model": "DistilBERT",
        "pretrained_model": MODEL_NAME,
        "source_csv": data.attrs["csv_path"],
        "text_column": text_col,
        "label_column": label_col,
        "label_mapping": data.attrs["label_mapping"],
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df), "seed": args.seed},
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "max_length": args.max_length, "learning_rate": args.learning_rate,
            "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "freeze_base": args.freeze_base, "device": str(device)
        },
        "training_time_seconds": training_seconds,
        "test_loss": test_loss,
        "metrics": metrics,
    }
    import json
    (OUTPUT_DIR / "distilbert_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="Validation loss")
    axes[0].set_title("DistilBERT loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["validation_f1"], marker="o", label="Validation F1")
    axes[1].set_title("Validation F1 by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distilbert_training_history.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham (0)", "Spam (1)"], yticklabels=["Ham (0)", "Spam (1)"])
    plt.title("DistilBERT — Test confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distilbert_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\nTest metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
          f"recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Saved model, metrics and plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
