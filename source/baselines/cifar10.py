import numpy as np
import seaborn as sns
import os
import time
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Loading function from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    """Load a pickled file."""
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

class Cifar10(Dataset):
    def __init__(self, data_folder='../data/cifar-10-batches-py', normalization='z-score', verbose=True,
                 plot_folder='../../plots/baselines'):
        """
        Initialize the CIFAR-10 dataset.

        :param data_folder: Path to the CIFAR-10 data folder.
        :param normalization: Normalization method ('z-score' or 'min-max').
        :param verbose: If True, prints dataset details.
        """
        self.data_folder = data_folder
        self.plot_folder = plot_folder
        self.verbose = verbose
        self.normalization = normalization

        start_time = time.time()

        # Load and process data
        self.images, self.labels = self._load_training_data()
        self.images_test, self.labels_test = self._load_test_data()
        self.label_names = self._load_label_names()

        # Compute statistics
        self.mean, self.std = self._compute_statistics(self.images)
        self.test_mean, self.test_std = self._compute_statistics(self.images_test)

        # Normalize data
        self._normalize_data()

        end_time = time.time()

        if verbose:
            self._print_dataset_summary(end_time - start_time)

    def _load_training_data(self):
        """Load and concatenate CIFAR-10 training batches."""
        images, labels = [], []
        for b in range(5):
            load_path = os.path.join(self.data_folder, f'data_batch_{b+1}')
            batch = unpickle(load_path)
            images.append(batch[b'data'])
            labels.append(batch[b'labels'])
        images = np.concatenate(images)
        labels = np.concatenate(labels)
        return self._reshape_images(images), labels

    def _load_test_data(self):
        """Load the CIFAR-10 test batch."""
        test_batch = unpickle(os.path.join(self.data_folder, 'test_batch'))
        images = self._reshape_images(test_batch[b'data'])
        labels = np.array(test_batch[b'labels'])
        return images, labels

    def _load_label_names(self):
        """Load label names from the metadata file."""
        load_path = os.path.join(self.data_folder, 'batches.meta')
        label_names = unpickle(load_path)[b'label_names']
        return [b.decode('utf-8') for b in label_names]

    @staticmethod
    def _reshape_images(data):
        """Reshape CIFAR-10 data into HWC format."""
        red = data[:, :1024].reshape((-1, 32, 32))
        green = data[:, 1024:2048].reshape((-1, 32, 32))
        blue = data[:, 2048:].reshape((-1, 32, 32))
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)

    @staticmethod
    def _compute_statistics(data):
        """Compute mean and standard deviation for the dataset."""
        return np.mean(data), np.std(data)

    def _normalize_data(self):
        """Normalize training and test data based on the selected method."""
        if self.normalization == 'z-score':
            self.images = (self.images - self.mean) / self.std
            self.images_test = (self.images_test - self.test_mean) / self.test_std
        elif self.normalization == 'min-max':
            self.images = (self.images - np.min(self.images)) / (np.max(self.images) - np.min(self.images))
            self.images_test = (self.images_test - np.min(self.images_test)) / (np.max(self.images_test) - np.min(self.images_test))

    def _print_dataset_summary(self, load_time):
        """Print a summary of the dataset."""
        print('-------------------------')
        print(f'CIFAR-10 loaded successfully in {load_time:.2f}s')
        print(f'Size: images-> {self.images.shape} \t labels -> {self.labels.shape}')
        print(f'Unique Labels: {np.unique(self.labels)}')
        print(f'Label Names: {self.label_names}')
        print(f'Mean: {self.mean} Std: {self.std}')
        print('-------------------------\n')

    def get_train_test_split(self):
        """Return training and test data splits."""
        return self.images, self.labels, self.images_test, self.labels_test

    def get_number_of_classes(self):
        """Return the number of unique classes."""
        return len(self.label_names)

    def statistical_analysis(self):
        """
        Perform statistical analysis on the dataset, including class distribution,
        pixel value distribution, and visualization of random samples.
        """
        self._plot_class_distribution()
        self._plot_pixel_distribution()
        self._plot_image_grid()
        print(f'Statistical analysis completed. Plots saved to {self.plot_folder}')

    def _plot_class_distribution(self):
        """Plot the class distribution of the dataset."""
        _, class_counts = np.unique(self.labels, return_counts=True)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=self.label_names, y=class_counts, palette='viridis')
        plt.title('Class Distribution')
        plt.xlabel('Class Label')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder, 'class_distribution.png'))
        plt.close()

    def _plot_pixel_distribution(self):
        """Plot the pixel value distribution after normalization."""
        flattened_pixels = self.images.flatten()
        sampled_pixels = np.random.choice(flattened_pixels, size=100000, replace=False)
        plt.figure(figsize=(8, 6))
        sns.histplot(sampled_pixels, bins=50, kde=True, color='green')
        plt.title('Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder, 'pixel_distribution.png'))
        plt.close()

    def _plot_image_grid(self):
        """Display a random grid of images."""
        random_indices = np.random.choice(len(self.images), size=16, replace=False)
        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(random_indices):
            plt.subplot(4, 4, i + 1)
            temp_img = self.visualize(idx)
            plt.imshow(temp_img.astype(np.uint8))
            plt.title(self.label_names[self.labels[idx]])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder, 'image_grid.png'))
        plt.close()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.labels.shape[0]

    def __getitem__(self, index):
        """Retrieve a single sample by index."""
        return self.images[index], self.labels[index]

    def visualize(self, idx):
        """
        Convert a CIFAR-10 data sample into a visualized image.

        :param idx: Index of the sample to visualize.
        :return: RGB image as a numpy array.
        """
        image = self.images[idx]
        if self.normalization == 'z-score':
            image = image * self.std + self.mean
        elif self.normalization == 'min-max':
            image = image * (np.max(image) - np.min(image)) + np.min(image)
        return image
