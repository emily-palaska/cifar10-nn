import numpy as np

class SVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        for i in range(self.n_iters):
            print(f"\rProgress: {100 * i / self.n_iters : .2f}%", end='')
            for idx, x_i in enumerate(x):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, x):
        linear_output = np.dot(x, self.w) - self.b
        return np.sign(linear_output)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"\rAccuracy: {accuracy * 100:.2f}%")
        return accuracy