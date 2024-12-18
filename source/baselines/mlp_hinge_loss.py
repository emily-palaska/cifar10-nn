from .mlp_utils import *
import time

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

        # Placeholders
        self.Z1 = None
        self.Z2 = None
        self.A1 = None

    def forward(self, x):
        # Modify dimensions if needed
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        # Forward pass
        self.Z1 = np.dot(x, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2  # Logits

    def backward(self, x, y_true, logits, learning_rate):
        # Modify dimensions if needed
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        # Backpropagation
        y_pred = logits
        loss_grad = hinge_loss_gradient(y_true, y_pred)

        # Gradients for output layer
        dz2 = loss_grad
        dw2 = np.dot(self.A1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradients for hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.Z1)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1


# Training the MLP with One-vs-One strategy
def train_one_vs_one(x, y, input_dim, hidden_dim, num_classes, epochs=10, learning_rate=0.01):
    pairs = one_vs_one_pairs(num_classes)
    classifiers = {}
    results = {}

    for i, j in pairs:
        print(f"Training classifier for classes {i} vs {j}")
        # Prepare binary labels
        mask = (y == i) | (y == j)
        x_pair = x[mask]
        y_pair = np.where(y[mask] == i, 1, -1)

        # Create and train an MLP for this pair
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)
        loss = []
        duration = []
        for epoch in range(epochs):
            start_time = time.time()
            logits = model.forward(x_pair)
            loss.append(hinge_loss(y_pair[:, None], logits))
            model.backward(x_pair, y_pair[:, None], logits, learning_rate)
            end_time = time.time()
            duration.append(end_time - start_time)
        results[f'{i}_vs_{j}'] = {'loss': loss, 'duration': duration}
        classifiers[(i, j)] = model

    return classifiers, results


# Prediction
def predict_one_vs_one(classifiers, x_test, y_test):
    votes = np.zeros((x_test.shape[0], 10))

    for (i, j), model in classifiers.items():
        logits = model.forward(x_test)
        predictions = np.sign(logits).flatten()
        votes[:, i] += (predictions == 1)
        votes[:, j] += (predictions == -1)

    y_pred = np.argmax(votes, axis=1)
    return metrics(y_pred, y_test)


