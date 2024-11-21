import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# ResNet18 Implementation
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Simplified ResNet18 architecture
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "metrics": {}}

    def train(self, epochs):
        self.model.to(self.device)
        for epoch in range(epochs):
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
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

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
                preds = torch.argmax(outputs, dim=1)  # Get predicted class indices
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())  # No need for torch.argmax here

        # Calculate metrics
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
