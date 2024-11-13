import numpy as np

from cifar10 import Cifar10, unpickle
from knn import KNNClassifier
from nearest_centroid import NearestCentroidClassifier
from evaluate import evaluate

# Step 1: Load the CIFAR-10 dataset
dataset = Cifar10(normalization='min-max')

# Step 2: Prepare the dataset
X_train = dataset.images
y_train = dataset.labels

# Step 3: Load the test batch and normalize it with z-score
test_batch = unpickle('./data/cifar-10-batches-py/test_batch')
X_test = np.array(test_batch[b'data'])
# normalization
#X_test = (X_test - np.mean(X_test)) / np.std(X_test)
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
y_test = np.array(test_batch[b'labels'])

# Step 4: Fit the KNN model
k=1
nc = NearestCentroidClassifier()
nc.fit(X_train, y_train)
# add scatter plot

# Step 5: Predict on the entire test set
y_pred = []
for i in range(len(X_test)):
    print(f'\rprogress: {100 * i / len(X_test) : .2f}%', end='')
    prediction = nc.predict(X_test[i].reshape(1, -1))
    y_pred.append(prediction)
print('\rPredictions completed')

# Convert predictions to numpy array for easier handling
y_pred = np.array(y_pred)

# Step 6: Evaluate the predictions
class_labels = dataset.label_names
evaluate(y_test, y_pred, class_labels, method='NC-Pair', k=k)


