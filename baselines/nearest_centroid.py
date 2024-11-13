import numpy as np
from sklearn.metrics import pairwise_distances
from skimage.metrics import structural_similarity as ssim


class NearestCentroidClassifier:
    def __init__(self):
        self.centroids = {}
        self.labels = None

    def fit(self, features, labels):
        """
        Fit the Nearest Centroid model by calculating the centroid of each class.
        :param features: NumPy array of shape (num_samples, num_features)
        :param labels: NumPy array of shape (num_samples,)
        """
        self.labels = np.unique(labels)
        for label in self.labels:
            # Select features that belong to the current label
            class_features = features[labels == label]
            # Compute the centroid (mean) of the class
            centroid = np.mean(class_features, axis=0)
            self.centroids[label] = centroid

    def predict(self, new_feature, method='pairwise_distance'):
        """
        Predict the label of a new sample based on the nearest centroid.
        :param new_feature: NumPy array of shape (1, num_features)
        :param method: string that specifies the distance metric
        :return: Predicted label
        """
        distances = []

        if method == 'pairwise_distance':
            new_feature = new_feature.flatten()  # Flatten image if needed
            for label, centroid in self.centroids.items():
                distance = np.linalg.norm(new_feature - centroid.flatten())
                distances.append((label, distance))

        elif method == 'mse':
            new_feature = new_feature.flatten()
            for label, centroid in self.centroids.items():
                distance = np.mean((centroid.flatten() - new_feature) ** 2)
                distances.append((label, distance))

        elif method == 'structural_similarity':
            new_feature = new_feature.reshape(32, 32, 3)
            for label, centroid in self.centroids.items():
                centroid_image = centroid.reshape(32, 32, 3)
                data_range = np.max(new_feature) - np.min(new_feature)
                score= ssim(new_feature, centroid_image, channel_axis=2, data_range=data_range)
                distance = 1 - score  # Invert SSIM for distance metric
                distances.append((label, distance))

        elif method == 'cosine_similarity':
            new_feature = new_feature.flatten()
            for label, centroid in self.centroids.items():
                cosine_distance = 1 - np.dot(new_feature, centroid.flatten()) / (
                        np.linalg.norm(new_feature) * np.linalg.norm(centroid.flatten())
                )
                distances.append((label, cosine_distance))

        else:
            print('Specified Nearest Centroid method is not supported.')
            return None

        # Step 2: Select the label with the minimum distance
        predicted_label = min(distances, key=lambda x: x[1])[0]
        return predicted_label
