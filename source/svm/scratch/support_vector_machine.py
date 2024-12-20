import numpy as np
from .utils import metrics
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
        if self.gamma is None:
            self.gamma = 1 / self.n_features

        if kernel == "linear":
            return lambda x, y: np.dot(x, y.T)
        elif kernel == "polynomial":
            return lambda x, y: (self.gamma * np.dot(x, y.T) + self.coef0) ** self.degree
        elif kernel == "rbf":
            def rbf_kernel(x, y):
                x_norm = np.sum(x ** 2, axis=1)[:, np.newaxis]  # Shape: (n_samples_x, 1)
                y_norm = np.sum(y ** 2, axis=1)[np.newaxis, :]  # Shape: (1, n_samples_y)
                distances = x_norm + y_norm - 2 * np.dot(x, y.T)
                return np.exp(-self.gamma * distances)

            return rbf_kernel
        elif kernel == "sigmoid":
            return lambda x, y: np.tanh(self.gamma * np.dot(x, y.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def fit(self, x, y):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)
        self.x = x
        self.y = y_
        self.alpha = np.zeros(n_samples)
        self.kernel_matrix = self.kernel(x, x)

        loss = []
        duration = []

        for i in range(self.n_iters):
            start_time = time.time()
            print(f"\rProgress: {100 * i / self.n_iters : .2f}%", end='')

            for j in range(n_samples):
                margin = np.sum(self.alpha * y_ * self.kernel_matrix[:, j]) - self.b
                if y_[j] * margin < 1:
                    self.alpha[j] += self.lr
                    self.b -= self.lr * y_[j]
                else:
                    self.alpha[j] -= self.lr * self.lambda_param * self.alpha[j]

            hinge_loss = 1 - y_ * (np.dot(self.alpha * y_, self.kernel_matrix) - self.b)
            loss.append(np.mean(np.maximum(0, hinge_loss)) + self.lambda_param * np.dot(self.alpha, self.alpha))
            end_time = time.time()
            duration.append(end_time - start_time)
        for l in loss:
            print('\r',l)
        return {'loss': loss, 'duration': duration}

    def predict(self, x):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        decision_values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            kernel_values = self.kernel(self.x, x[i:i + 1])
            decision_values[i] = np.sum(self.alpha * self.y * kernel_values) - self.b

        return np.sign(decision_values)

    def evaluate(self, x_test, y_test):
        if x_test.ndim != 2:
            x_test = x_test.reshape(x_test.shape[0], -1)

        y_pred = self.predict(x_test)
        return metrics(y_test, y_pred, verbose=False)

    def visualize(self, save_dir="../plots/svm/"):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Downsample if necessary
        if self.x.shape[0] > 10000:
            indices = np.random.choice(self.x.shape[0], 1000, replace=False)
            x_vis = self.x[indices]
            y_vis = self.y[indices]
            alpha_vis = self.alpha[indices]
        else:
            x_vis = self.x
            y_vis = self.y
            alpha_vis = self.alpha

        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_x = tsne.fit_transform(x_vis)

        # Create a mesh grid for the decision boundary in 2D t-SNE space
        x_min, x_max = reduced_x[:, 0].min() - 1, reduced_x[:, 0].max() + 1
        y_min, y_max = reduced_x[:, 1].min() - 1, reduced_x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Compute decision values for the grid in t-SNE space
        decision_values = np.zeros(grid_points.shape[0])
        for i in range(grid_points.shape[0]):
            distances = np.sum((reduced_x - grid_points[i]) ** 2, axis=1)
            kernel_values = np.exp(-self.gamma * distances)
            decision_values[i] = np.sum(alpha_vis * y_vis * kernel_values) - self.b

        decision_values = decision_values.reshape(xx.shape)
        print(np.min(decision_values), np.max(decision_values))

        # Plot decision boundary and margins
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, decision_values, levels=50, cmap="coolwarm", alpha=0.3)
        plt.contour(xx, yy, decision_values, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
                    linestyles=['--', '-', '--'], linewidths=1)

        # Scatter plot of the data points
        plt.scatter(reduced_x[:, 0], reduced_x[:, 1], c=y_vis, cmap='coolwarm', alpha=0.7, edgecolor='k')
        plt.title(f"SVM Decision Boundary Visualization with {self.kernel_type} kernel")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Decide file name to avoid overwriting
        idx = 0
        while os.path.exists(os.path.join(save_dir, f"svm_visualization_{self.kernel_type}{idx}.png")):
            idx += 1

        # Save plot
        save_path = os.path.join(save_dir, f"svm_visualization_{self.kernel_type}{idx}.png")
        plt.savefig(save_path)