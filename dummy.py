import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# Function to load JSON data
def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


# Function to extract confusion matrix and display it
def plot_confusion_matrix(json_data):
    try:
        # Extract confusion matrix from 'test_metrics'
        confusion_matrix = json_data['test_metrics']['confusion_matrix']
        confusion_matrix = np.array(confusion_matrix)

        # Plot confusion matrix using ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        disp.plot(cmap=plt.cm.Blues, colorbar=True)

        plt.title("Confusion Matrix")
        plt.show()
    except KeyError as e:
        print(f"KeyError: {e}. Please ensure the JSON file contains 'test_metrics' and 'confusion_matrix'.")


# Main script
if __name__ == "__main__":
    # Replace 'path_to_file.json' with the path to your JSON file
    #file_path = "training_results.json"
    file_path = 'training_results.json'

    # Load JSON file
    json_data = load_json(file_path)

    # Plot confusion matrix
    plot_confusion_matrix(json_data)
