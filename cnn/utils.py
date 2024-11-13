import numpy as np

class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_output):
        return d_output * (self.last_input > 0)

class Softmax:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

def cross_entropy_loss(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def train(x_train, y_train, model, epochs=10, lr=0.01):
    for epoch in range(epochs):
        loss = 0
        correct = 0

        for x, y in zip(x_train, y_train):
            # Forward pass
            output = model.forward(x)
            loss += cross_entropy_loss(output, y)
            if np.argmax(output) == np.argmax(y):
                correct += 1

            # Backward pass
            grad = model.backward(y)
            model.update_weights(lr)

        print(f"Epoch {epoch+1}, Loss: {loss / len(x_train)}, Accuracy: {correct / len(x_train)}")