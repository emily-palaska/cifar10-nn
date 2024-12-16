import numpy as np
from decorator import append

from .utils import metrics, append_to_json
from .svm_classifier import SVMClassifier

class OneVsAllClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, results_file="one_vs_all_results.json"):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.results_file = results_file
        self.models = {}  # To store individual classifiers for each class

    def fit(self, x_train, y_train, n_classes):
        """
        Train one SVM classifier per class using the one-vs-all strategy.
        """
        results = {}
        for class_idx in range(n_classes):
            print(f"\rTraining classifier for class {class_idx} vs all...")
            y_binary = np.where(y_train == class_idx, 1, 0)  # Convert labels to 1 vs all
            svm = SVMClassifier(
                learning_rate=self.lr,
                lambda_param=self.lambda_param,
                n_iters=self.n_iters
            )
            results[f'{class_idx}'] = svm.fit(x_train, y_binary)
            self.models[class_idx] = svm

        append_to_json(self.results_file, {'fit': results})
        print("\r                       ")

    def predict(self, x):
        """
        Predict the class of each sample by selecting the classifier with the highest score.
        """
        predictions = []
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        for class_idx, model in self.models.items():
            linear_output = np.dot(x, model.w) - model.b
            predictions.append(linear_output)
        predictions = np.stack(predictions, axis=1)
        return np.argmax(predictions, axis=1)

    def evaluate(self, x_test, y_test, n_classes):
        """
        Evaluate the performance of the one-vs-all classifier.
        """
        # Evaluation of each model separately
        evaluation_metrics = {}
        for class_idx in range(n_classes):
            y_binary = np.where(y_test == class_idx, 1, 0)  # Convert labels to 1 vs all
            evaluation_metrics[f'{class_idx}'] = self.models[class_idx].evaluate(x_test, y_binary)

        # Overall evaluation
        y_pred = self.predict(x_test)
        evaluation_metrics['general'] = metrics(y_pred, y_test)
        append_to_json(self.results_file, {"test": evaluation_metrics})
        return evaluation_metrics
