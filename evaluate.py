import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate(y_test, y_pred, class_labels, method, k, output_directory='./'):
    # Calculate accuracy, precision, recall, and F-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {f_score:.4f}")

    # Create and save the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=class_labels)
    plot_file_path = os.path.join(output_directory, f'confusion_matrix_{method}_{k}.png')
    disp.plot()
    plt.title(f'Confusion Matrix using {method} with k={k}')
    plt.xticks(rotation=45)
    plt.savefig(plot_file_path)