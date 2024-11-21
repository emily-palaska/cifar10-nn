import torch
from ready_models.resnet18 import Trainer, ResNet18
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from baselines.cifar10 import Cifar10
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization="min-max")

    x_train = dataset.images.reshape((dataset.images.shape[0], 3, 32, 32))
    y_train = dataset.labels
    x_test = dataset.images_test.reshape((dataset.images_test.shape[0], 3, 32, 32))
    y_test = dataset.labels_test

    # Split train into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train
    )

    # Convert to tensors
    x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

    # Define model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train and evaluate
    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)
    trainer.train(epochs=20)

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, dataset_type="test")

    # Save metrics
    trainer.history["metrics"] = test_metrics
    trainer.save_metrics()

    print('Results saved in metrics.json')
