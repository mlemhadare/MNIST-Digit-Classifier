"""
train.py
Training pipeline with Optuna hyperparameter tuning and MLflow tracking.
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MNISTCNN
from data import get_data_loaders
import mlflow
from mlflow import pytorch
import optuna
import numpy as np
import os
from utils import plot_confusion_matrix, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    Returns average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_data_loaders("data", batch_size)
    model = MNISTCNN(dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=3)
    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if early_stopping(val_loss):
            break
    return val_loss


def run_optuna_study() -> Dict[str, Any]:
    """
    Run Optuna study and return best hyperparameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_params


def train_with_mlflow(params: Dict[str, Any]) -> None:
    """
    Train final model with MLflow tracking and save artifacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders("data", params["batch_size"])
    model = MNISTCNN(dropout_rate=params["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    early_stopping = EarlyStopping(patience=3)
    mlflow.set_experiment("MNIST-Classification")
    with mlflow.start_run():
        mlflow.log_params(params)
        best_val_loss = float("inf")
        for epoch in range(20):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, step=epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
            if early_stopping(val_loss):
                break
        # Evaluate on test set
        model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, "confusion_matrix.png")
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        if isinstance(f1, np.ndarray):
            f1 = f1.mean()  
        mlflow.log_metric("test_f1", float(f1))  
        mlflow.log_artifact("confusion_matrix.png")
        pytorch.log_model(model, "model")
        mlflow.log_artifact("best_model.pth")

if __name__ == "__main__":
    best_params = run_optuna_study()
    train_with_mlflow(best_params)
