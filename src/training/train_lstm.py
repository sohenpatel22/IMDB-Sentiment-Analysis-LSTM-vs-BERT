import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.plotting import plot_curves
from src.data.preprocess import load_data, prepare_lstm_data
from src.data.dataset import LSTMDataset
from src.models.lstm_model import SentimentRNN
from src.training.evaluate import evaluate_lstm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()

        correct += (preds == labels.int()).sum().item()
        total += labels.size(0)

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
    data_dict, stoi = prepare_lstm_data(df, vocab_size=5000, seq_length=500)

    train_dataset = LSTMDataset(data_dict["X_train"], data_dict["y_train"])
    val_dataset = LSTMDataset(data_dict["X_val"], data_dict["y_val"])
    test_dataset = LSTMDataset(data_dict["X_test"], data_dict["y_test"])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SentimentRNN(vocab_size=len(stoi), embedding_dim=128, hidden_dim=128).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    best_val_acc = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_lstm(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 50)

        save_checkpoint("checkpoints/lstm_last.pt", model, optimizer, epoch, val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint("checkpoints/lstm_best.pt", model, optimizer, epoch, best_val_acc)
            print("Best model saved.")

    plot_curves(history, save_prefix="outputs/figures/lstm")

    best_model = SentimentRNN(vocab_size=len(stoi), embedding_dim=128, hidden_dim=128).to(device)
    load_checkpoint("checkpoints/lstm_best.pt", best_model, device=device)

    test_metrics = evaluate_lstm(best_model, test_loader, device)
    print("Test Metrics:", test_metrics)

    with open("outputs/results/lstm_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()