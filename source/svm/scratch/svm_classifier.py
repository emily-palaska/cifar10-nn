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
            if self.gamma is None:
                self.gamma = 1 / self.n_features  # Default gamma
    
            def rbf_kernel(x, y):
                # Pairwise computation for each row in x
                if x.ndim == 1:  # Single sample in x
                    distances = np.sum(x**2) + np.sum(y**2, axis=1) - 2 * np.dot(y, x)
                    return np.exp(-self.gamma * distances)
                else:  # Multiple samples in x
                    kernel_values = np.zeros((x.shape[0], y.shape[0]))
                    for i in range(x.shape[0]):
                        distances = np.sum(x[i]**2) + np.sum(y**2, axis=1) - 2 * np.dot(y, x[i])
                        kernel_values[i] = np.exp(-self.gamma * distances)
                    return kernel_values
    
            return rbf_kernel
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

        # Compute the decision boundary using the kernel
        decision_values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            kernel_values = self.kernel(self.x, x[i:i + 1])  # Compute kernel values
            decision_values[i] = np.sum(self.alpha * self.y * kernel_values) - self.b

        # Return the sign of the decision boundary as predictions
        return np.sign(decision_values)

    def evaluate(self, x_test, y_test):
        if x_test.ndim != 2:
            x_test = x_test.reshape(x_test.shape[0], -1)

        # Get predictions
        y_pred = self.predict(x_test)

        # Compute evaluation metrics using the utility function
        return metrics(y_test, y_pred, verbose=False)
