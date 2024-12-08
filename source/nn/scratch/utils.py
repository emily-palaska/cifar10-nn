import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_output):
        return d_output * (self.last_input > 0)

class Softmax:
    def forward(self, x):
        # Clip input values to avoid overflow in np.exp
        x_clipped = np.clip(x, -700, 700)  # np.exp can handle values in this range without overflow
        exp_values = np.exp(x_clipped - np.max(x_clipped, axis=1, keepdims=True))
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

def metrics(predictions, true_labels):
    acc = accuracy_score(true_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

    print("Evaluation Results on Test dataset:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {fscore:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": fscore,
        "confusion_matrix": conf_matrix.tolist()
    }
