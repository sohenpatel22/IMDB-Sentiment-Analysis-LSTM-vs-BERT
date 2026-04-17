import os
import json
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.plotting import plot_curves
from src.data.preprocess import load_data
from src.data.dataset import MovieReviewDataset
from src.data.tokenizer_utils import get_tokenizer
from src.models.bert_classifier import SentimentClassifier
from src.training.evaluate import calculate_metrics


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    df = load_data("data/raw/IMDB Dataset.csv")

    X = df["review"].values
    y = df["sentiment"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    tokenizer = get_tokenizer()

    train_dataset = MovieReviewDataset(X_train, y_train, tokenizer, max_len=400)
    val_dataset = MovieReviewDataset(X_val, y_val, tokenizer, max_len=400)
    test_dataset = MovieReviewDataset(X_test, y_test, tokenizer, max_len=400)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SentimentClassifier(n_classes=2).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_val_acc = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = calculate_metrics(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 50)

        save_checkpoint("checkpoints/bert_last.pt", model, optimizer, epoch, val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint("checkpoints/bert_best.pt", model, optimizer, epoch, best_val_acc)
            print("Best BERT model saved.")

    plot_curves(history, save_prefix="outputs/figures/bert")

    best_model = SentimentClassifier(n_classes=2).to(device)
    load_checkpoint("checkpoints/bert_best.pt", best_model, device=device)

    test_metrics = calculate_metrics(best_model, test_loader, device)
    print("Test Metrics:", test_metrics)

    with open("outputs/results/bert_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()