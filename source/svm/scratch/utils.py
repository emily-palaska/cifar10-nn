import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

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

def append_to_json(file_path, new_data):
    try:
        # Open the existing JSON file and load its contents
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty dictionary
        data = {}

    # Update the dictionary with the new data
    data.update(new_data)

    # Save the updated dictionary back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)