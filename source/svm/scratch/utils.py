import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def one_vs_all(labels, target):
    return np.where(labels == target, 1, -1)

def metrics(predictions, true_labels):
    acc = accuracy_score(true_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

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
