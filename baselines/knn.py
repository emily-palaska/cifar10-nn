import numpy as np
from sklearn.metrics import pairwise_distances
from skimage.metrics import structural_similarity as ssim

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.features = None # assume this is many 256x256 images
        self.labels = None
    
    def fit(self, features, labels):
        """
        Fit the KNN model by storing the training features and labels.
        :param features: NumPy array of shape (num_samples, num_features)
        :param labels: NumPy array of shape (num_samples,)
        """
        self.features = features
        self.labels = labels
    
    def predict(self, new_feature, method='pairwise_distance'):
        """
        Predict the label of a new sample based on the k nearest neighbors.
        :param method: string that takes values: pairwise_distance (default), mse, structural_similarity, cosine_similarity
        :param new_feature: NumPy array of shape (1, num_features)
        :return: Predicted label
        """
        # Step 1: Compute the distance between new_feature and all other features

        if method == 'pairwise_distance':
            self. features = self.features.reshape(self.features.shape[0], -1)  # Flatten each image (256x256) -> (65536,)
            distances = pairwise_distances(new_feature, self.features, metric='euclidean').flatten()
        elif method == 'mse':
            self.features = self.features.reshape(self.features.shape[0], -1)  # Flatten images
            distances = np.mean((self.features - new_feature) ** 2, axis=1)
        elif method == 'structural_similarity':
            self.features = self.features.reshape(self.features.shape[0], 32, 32, 3)  # Reshape to original image shape
            new_feature = new_feature.reshape(32, 32, 3)  # Reshape to original image shape
            distances = []
            for feature in self.features:
                data_range = np.max(new_feature) - np.min(new_feature)
                score = ssim(new_feature, feature, channel_axis=2, data_range=data_range)
                distances.append(1 - score)  # Invert SSIM for distance metric
            distances = np.array(distances)
        elif method == 'cosine_similarity':
            self.features = self.features.reshape(self.features.shape[0], -1)  # Flatten images
            distances = 1 - pairwise_distances(new_feature, self.features, metric='cosine').flatten()
        else:
            print('Specified KNN method is not supported.')
            return None
        # Step 2: Find the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Step 3: Find the labels of the k-nearest neighbors
        k_labels = self.labels[k_indices]
        
        # Step 4: Return the most common label among the neighbors
        return np.bincount(k_labels).argmax()