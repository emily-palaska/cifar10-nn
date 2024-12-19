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
                if x.ndim == 1:
                    distances = np.sum(x**2) + np.sum(y**2, axis=1) - 2 * np.dot(y, x)
                    return np.exp(-self.gamma * distances)
                else:
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

        # Downsample to 1000 samples if necessary
        if self.x.shape[0] > 1000:
            indices = np.random.choice(self.x.shape[0], 1000, replace=False)
            x_vis = self.x[indices]
            y_vis = self.y[indices]
        else:
            x_vis = self.x
            y_vis = self.y

        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_x = tsne.fit_transform(x_vis)

        # Plot decision boundary and points
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_x[:, 0], reduced_x[:, 1], c=y_vis, cmap='coolwarm', alpha=0.7, edgecolor='k')
        #TODO line
        """
        # Calculate the decision boundary in the original feature space (before t-SNE projection)
        # Use the support vectors to approximate the margin
        support_vectors = self.x[self.alpha > 0]
        support_vector_labels = self.y[self.alpha > 0]

        # Compute the projections of the support vectors in the reduced space (t-SNE)
        tsne_support_vectors = tsne.fit_transform(support_vectors)

        # Calculate the weights of the decision boundary in the feature space
        w = self.w
        b = self.b

        # We approximate the decision boundary using the transformed support vectors
        # The decision boundary in 2D can be approximated using w and b
        # We'll plot a line between the support vectors closest to the decision boundary

        # Assuming the decision boundary is along the line connecting the support vectors
        min_sv = tsne_support_vectors[np.argmin(np.linalg.norm(tsne_support_vectors, axis=1))]
        max_sv = tsne_support_vectors[np.argmax(np.linalg.norm(tsne_support_vectors, axis=1))]

        # Plot the line (approximately the maximum margin line)
        plt.plot([min_sv[0], max_sv[0]], [min_sv[1], max_sv[1]], 'k--', label="Max Margin")
        """
        plt.title("SVM Decision Boundary Visualization")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Decide file name to avoid overwriting
        idx = 0
        while os.path.exists(os.path.join(save_dir, f"svm_visualization{idx}.png")):
            idx += 1

        # Save plot
        save_path = os.path.join(save_dir, f"svm_visualization{idx}.png")
        plt.savefig(save_path)
        plt.show()
