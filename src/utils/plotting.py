import os
import matplotlib.pyplot as plt


def plot_curves(history, save_prefix=None):
    required_keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Missing key in history: {key}")

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()

    if save_prefix is not None:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True) if os.path.dirname(save_prefix) else None
        plt.savefig(f"{save_prefix}_loss.png", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_accuracy.png", bbox_inches="tight")
    plt.show()
    plt.close()