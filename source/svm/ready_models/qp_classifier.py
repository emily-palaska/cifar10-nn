import numpy as np
import cvxopt, os, json
from .utils import metrics

class QPClassifier:
    def __init__(self, c=1.0, max_samples=50000):
        self.c = c
        self.max_samples = max_samples
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.w = None
        self.b = None

    def fit(self, x, y):
        # Reshape input data if necessary
        if x.ndim != 2:
            x = x.reshape((x.shape[0], -1))

        # Limit the number of samples to max_samples
        if len(x) > self.max_samples:
            indices = np.random.choice(len(x), self.max_samples, replace=False)
            x = x[indices]
            y = y[indices]

        n_samples, n_features = x.shape
        y = np.where(y == 0, -1, 1).astype(np.float64)  # Ensure dtype is float64

        # Compute the kernel matrix (linear kernel)
        k = np.dot(x, x.T)

        # Define the quadratic programming problem
        p = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-np.ones(n_samples))
        g = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.c)))
        a = cvxopt.matrix(y.reshape(1, -1))  # Shape (1, n_samples)
        b = cvxopt.matrix(0.0)

        # Solve the QP problem
        solution = cvxopt.solvers.qp(p, q, g, h, a, b)

        self.alpha = np.array(solution['x']).flatten()
        support_vector_indices = self.alpha > 1e-5
        self.alpha = self.alpha[support_vector_indices]
        self.support_vectors = x[support_vector_indices]
        self.support_labels = y[support_vector_indices]

        self.w = np.sum(self.alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)
        self.b = np.mean(self.support_labels - np.dot(self.support_vectors, self.w))

    def predict(self, X):
        decision = np.dot(X, self.w) + self.b
        return np.sign(decision)

    def evaluate(self, x_test, y_test, file_name='qp.json', save_dir='../results/svm/'):
        if x_test.ndim != 2:
            x_test = x_test.reshape((x_test.shape[0], -1))

        y_pred = self.predict(x_test)
        evaluation_metrics = metrics(y_pred, y_test)

        with open(os.path.join(save_dir, file_name), "w") as f:
            json.dump(evaluation_metrics, f, indent=4)

        return evaluation_metrics
