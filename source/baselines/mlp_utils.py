import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Define hinge loss
def hinge_loss(y_true, y_pred):
    # Hinge loss: max(0, 1 - y_true * y_pred)
    return np.maximum(0, 1 - y_true * y_pred).mean()

def hinge_loss_gradient(y_true, y_pred):
    # Gradient of hinge loss
    grad = np.zeros_like(y_pred)
    grad[y_true * y_pred < 1] = -y_true[y_true * y_pred < 1]
    return grad / len(y_true)

# One-vs-One strategy setup
def one_vs_one_pairs(num_classes):
    return [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]

def metrics(predictions, true_labels, verbose=True):
    acc = accuracy_score(true_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

    if verbose:
        print("Evaluation Results on Test dataset:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {fscore:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": fscore,
        "confusion_matrix": conf_matrix.tolist()
    }