import numpy as np
from .utils import metrics
import time

class SVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel="linear", degree=3, gamma=None,
                 coef0=1, n_features=3072):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_features = n_features
        self.kernel = self._get_kernel(kernel)

        # Placeholders
        self.w = None
        self.b = None
        self.x = None
        self.y = None
        self.alpha = None
        self.kernel_matrix = None


    def _get_kernel(self, kernel):
        # Default gamma = 1 / n_features
        if self.gamma is None:
            self.gamma = 1 / self.n_features

        if kernel == "linear":
            return lambda x, y: np.dot(x, y.T)
        elif kernel == "polynomial":
            return lambda x, y: (self.gamma * np.dot(x, y.T) + self.coef0) ** self.degree
        elif kernel == "rbf":
            return lambda x, y: np.exp(-self.gamma * np.linalg.norm(x[:, None] - y, axis=2) ** 2)
        elif kernel == "sigmoid":
            return lambda x, y: np.tanh(self.gamma * np.dot(x, y.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def fit(self, x, y):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # Weights are not used in kernelized SVM
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)
        self.x = x
        self.y = y_
        self.alpha = np.zeros(n_samples)  # Dual coefficients for kernel SVM
        self.kernel_matrix = self.kernel(x, x)

        loss = []
        duration = []

        for i in range(self.n_iters):
            start_time = time.time()
            print(f"\rProgress: {100 * i / self.n_iters : .2f}%", end='')

            # Compute hinge loss and update alpha
            for j in range(n_samples):
                margin = np.sum(self.alpha * y_ * self.kernel_matrix[:, j]) - self.b
                if y_[j] * margin < 1:
                    self.alpha[j] += self.lr
                    self.b -= self.lr * y_[j]
                else:
                    self.alpha[j] -= self.lr * self.lambda_param * self.alpha[j]

            # Loss calculation
            hinge_loss = 1 - y_ * (np.dot(self.alpha * y_, self.kernel_matrix) - self.b)
            loss.append(np.mean(np.maximum(0, hinge_loss)) + self.lambda_param * np.dot(self.alpha, self.alpha))
            end_time = time.time()
            duration.append(end_time - start_time)

        return {'loss': loss, 'duration': duration}

    def predict(self, x):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        kernel_output = np.sum(self.alpha * self.y[:, None] * self.kernel(self.x, x), axis=0) - self.b
        return np.sign(kernel_output)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        evaluation_metrics = metrics(y_test, y_pred, verbose=False)
        return {"metrics": evaluation_metrics}
