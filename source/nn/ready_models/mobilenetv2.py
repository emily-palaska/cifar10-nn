import os, time
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# MobileNetV2 Implementation
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(
            # Initial Convolution
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # Depthwise Separable Block
            self._inverted_residual(32, 16, 1, 1),
            self._inverted_residual(16, 24, 6, 2),
            self._inverted_residual(24, 24, 6, 1),
            self._inverted_residual(24, 32, 6, 2),
            self._inverted_residual(32, 32, 6, 1),
            self._inverted_residual(32, 32, 6, 1),
            self._inverted_residual(32, 64, 6, 2),
            self._inverted_residual(64, 64, 6, 1),
            self._inverted_residual(64, 64, 6, 1),
            self._inverted_residual(64, 64, 6, 1),
            self._inverted_residual(64, 96, 6, 1),
            self._inverted_residual(96, 96, 6, 1),
            self._inverted_residual(96, 96, 6, 1),
            self._inverted_residual(96, 160, 6, 2),
            self._inverted_residual(160, 160, 6, 1),
            self._inverted_residual(160, 160, 6, 1),
            self._inverted_residual(160, 320, 6, 1),
            # Final Convolution
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _inverted_residual(self, inp, oup, expand_ratio, stride):
        hidden_dim = inp * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        return nn.Sequential(*layers)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "metrics": {}, 'time': []}

    def train(self, epochs):
        self.model.to(self.device)
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            train_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = self.validate()
            end_time = time.time()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["time"].append(end_time - start_time)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {end_time - start_time:.4f}")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def evaluate(self, loader, dataset_type="test"):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        conf_matrix = confusion_matrix(all_labels, all_preds) if dataset_type == "test" else None

        metrics = {
            f"{dataset_type}_accuracy": accuracy,
            f"{dataset_type}_precision": precision,
            f"{dataset_type}_recall": recall,
            f"{dataset_type}_fscore": fscore,
        }
        if dataset_type == "test":
            metrics["confusion_matrix"] = conf_matrix.tolist()

        return metrics

    def save_metrics(self, filepath="metrics.json"):
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4)
