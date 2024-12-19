import numpy as np
from .utils import metrics, append_to_json
from .svm_classifier import SVMClassifier

class OneVsOneClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel="linear", degree=3, gamma=None,
                 coef0=1, n_features=3072, results_file='../results/svm/1v1.json'):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_features = n_features
        self.results_file = results_file
        self.models = {}  # To store classifiers for each pair of classes

    def fit(self, x_train, y_train, n_classes):
        """
        Train one SVM classifier per pair of classes using the one-vs-one strategy.
        """
        results = {}
        for class_a in range(n_classes):
            for class_b in range(class_a + 1, n_classes):
                print(f"\rTraining classifier for class {class_a} vs class {class_b}...")

                # Select data points belonging to the current pair of classes
                mask = (y_train == class_a) | (y_train == class_b)
                x_pair = x_train[mask]
                y_pair = y_train[mask]
                y_binary = np.where(y_pair == class_a, 1, -1)  # Convert labels to binary

                # Train SVM for this pair
                svm = SVMClassifier(
                    learning_rate=self.lr,
                    lambda_param=self.lambda_param,
                    n_iters=self.n_iters,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    n_features=self.n_features
                )

                results[f'{class_a}_vs_{class_b}'] = svm.fit(x_pair, y_binary)
                #svm.visualize()
                self.models[(class_a, class_b)] = svm
        print('\r\n')
        append_to_json(self.results_file, {'fit': results})

    def predict(self, x):
        """
        Predict the class of each sample using a majority voting scheme among all pairwise classifiers.
        """
        votes = np.zeros((x.shape[0], len(self.models)))  # Store votes for each class
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        pair_idx = 0
        for (class_a, class_b), model in self.models.items():
            # Compute decision scores
            linear_output = np.dot(x, model.w) - model.b
            predictions = np.where(linear_output >= 0, class_a, class_b)
            votes[:, pair_idx] = predictions
            pair_idx += 1

        # Determine the most voted class for each sample
        return np.apply_along_axis(lambda row: np.bincount(row.astype(int), minlength=len(self.models)).argmax(),
                                   axis=1, arr=votes)

    def evaluate(self, x_test, y_test, n_classes):
        """
        Evaluate the performance of the one-vs-one classifier.
        """
        y_pred = self.predict(x_test)
        evaluation_metrics = {}

        # Evaluation of each model separately
        for class_a in range(n_classes):
            for class_b in range(class_a + 1, n_classes):
                # Select data points belonging to the current pair of classes
                mask = (y_test == class_a) | (y_test == class_b)
                x_pair = x_test[mask]
                y_pair = y_test[mask]
                y_binary = np.where(y_pair == class_a, 1, -1)  # Convert labels to binary

                evaluation_metrics[f'{class_a}_vs_{class_b}'] = self.models[(class_a, class_b)].evaluate(x_pair, y_binary)

        # Overall evaluation
        evaluation_metrics['general'] = metrics(y_pred, y_test)
        append_to_json(self.results_file, {"test": evaluation_metrics})
        return evaluation_metrics
