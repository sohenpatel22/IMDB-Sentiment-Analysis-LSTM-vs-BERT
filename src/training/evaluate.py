import torch

from src.training.metrics import calculate_classification_metrics


@torch.no_grad()
def evaluate_lstm(model, loader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()

        y_true.extend(labels.long().cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = calculate_classification_metrics(y_true, y_pred)

    if criterion is not None:
        metrics["loss"] = total_loss / len(loader)

    return metrics


@torch.no_grad()
def evaluate_bert(model, loader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if criterion is not None:
            loss = criterion(outputs, targets)
            total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(targets.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = calculate_classification_metrics(y_true, y_pred)

    if criterion is not None:
        metrics["loss"] = total_loss / len(loader)

    return metrics