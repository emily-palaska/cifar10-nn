import numpy as np


# Define activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# Define hinge loss
def hinge_loss(y_true, y_pred):
    # Hinge loss: max(0, 1 - y_true * y_pred)
    return np.maximum(0, 1 - y_true * y_pred).mean()


def hinge_loss_gradient(y_true, y_pred):
    # Gradient of hinge loss
    grad = np.zeros_like(y_pred)
    grad[y_true * y_pred < 1] = -y_true[y_true * y_pred < 1]
    return grad / len(y_true)


# One-vs-One strategy setup
def one_vs_one_pairs(num_classes):
    return [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]


# Initialize the MLP
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2  # Logits

    def backward(self, X, y_true, logits, learning_rate):
        # Backpropagation
        y_pred = logits
        loss_grad = hinge_loss_gradient(y_true, y_pred)

        # Gradients for output layer
        dZ2 = loss_grad
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


# Training the MLP with One-vs-One strategy
def train_one_vs_one(X, y, input_dim, hidden_dim, num_classes, epochs=10, learning_rate=0.01):
    pairs = one_vs_one_pairs(num_classes)
    classifiers = {}

    for i, j in pairs:
        print(f"Training classifier for classes {i} vs {j}")
        # Prepare binary labels
        mask = (y == i) | (y == j)
        X_pair = X[mask]
        y_pair = np.where(y[mask] == i, 1, -1)

        # Create and train an MLP for this pair
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)
        for epoch in range(epochs):
            logits = model.forward(X_pair)
            loss = hinge_loss(y_pair[:, None], logits)
            model.backward(X_pair, y_pair[:, None], logits, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        classifiers[(i, j)] = model

    return classifiers


# Prediction
def predict_one_vs_one(classifiers, X):
    votes = np.zeros((X.shape[0], 10))

    for (i, j), model in classifiers.items():
        logits = model.forward(X)
        predictions = np.sign(logits).flatten()
        votes[:, i] += (predictions == 1)
        votes[:, j] += (predictions == -1)

    return np.argmax(votes, axis=1)


# Example usage
if __name__ == "__main__":
    # CIFAR-10-like input
    np.random.seed(42)
    X_train = np.random.randn(500, 3072)  # 500 samples, 3072 features
    y_train = np.random.randint(0, 10, 500)  # Random labels for 10 classes

    # Train
    num_classes = 10
    hidden_dim = 128
    classifiers = train_one_vs_one(X_train, y_train, input_dim=3072, hidden_dim=hidden_dim, num_classes=num_classes,
                                   epochs=5, learning_rate=0.01)

    # Predict
    X_test = np.random.randn(100, 3072)  # 100 test samples
    y_pred = predict_one_vs_one(classifiers, X_test)
    print("Predicted classes:", y_pred)
