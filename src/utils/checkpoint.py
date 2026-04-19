import os
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, best_val_acc=None, history=None):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict()
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if best_val_acc is not None:
        checkpoint["best_val_acc"] = best_val_acc

    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint