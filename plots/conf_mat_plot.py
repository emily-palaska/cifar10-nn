import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# loading function from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Load the JSON data
json_file = "../results/svm/sklearn_rbf.json"  # Replace with your JSON file path
with open(json_file, "r") as file:
    data = json.load(file)

# Extract the confusion matrix4
confusion_matrix = np.array(data["confusion_matrix"])
confusion_matrix = np.round(confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True), 2)

# Load labels decoder to strings
load_path = '../source/data/cifar-10-batches-py/batches.meta'
labels = unpickle(load_path)[b'label_names']
labels = [b.decode('utf-8') for b in labels]

# Display and save the confusion matrix
display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
display.plot(cmap="viridis", colorbar=True)

# Save the plot
plt.title("Confusion Matrix:444444 LibSvm with RBF Kernel")
plt.xticks(rotation=45)
output_file = "svm/confusion_matrix.png"
plt.tight_layout()
plt.savefig(output_file)
plt.show()

print(f"Confusion matrix saved as {output_file}")
