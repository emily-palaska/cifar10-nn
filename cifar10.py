import numpy as np
import seaborn as sns
import os, time

from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# loading function from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

class Cifar10(Dataset):
    def __init__(self, data_folder='./data/cifar-10-batches-py', normalization='z-score', verbose=False):
        self.data_folder = data_folder
        self.verbose = verbose
        self.normalization = normalization

        # Load the cifar-10 batches
        start_time = time.time()
        self.images = []
        self.labels = []
        for b in range(5):
            load_path = os.path.join(self.data_folder, f'data_batch_{b+1}')
            batch = unpickle(load_path)
            self.images.append(batch[b'data'])
            self.labels.append(batch[b'labels'])

        # # Load the test batch
        # load_path = os.path.join(self.data_folder, 'test_batch')
        # test_batch = unpickle(load_path)
        # self.test_images = test_batch[b'data']
        # self.test_labels = test_batch[b'labels']

        # Load labels decoder to strings
        load_path = os.path.join(self.data_folder, 'batches.meta')
        self.label_names = unpickle(load_path)[b'label_names']
        self.label_names = [b.decode('utf-8') for b in self.label_names]

        # Turn data into numpy arrays
        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

        # Calculate mean and std for normalization
        # self.test_mean = np.mean(self.test_images)
        # self.test_std = np.std(self.test_images)
        self.mean = np.mean(self.images)
        self.std = np.std(self.images)

        # Reshpae to HWC
        # Separate each color channel and reshape
        red_channel = self.images[:,:1024].reshape((self.images.shape[0], 32, 32))
        green_channel = self.images[:,1024:2048].reshape((self.images.shape[0], 32, 32))
        blue_channel = self.images[:,2048:].reshape((self.images.shape[0], 32, 32))

        # Stack the channels along the third dimension to create an RGB image
        self.images = np.stack((red_channel, green_channel, blue_channel), axis=-1).astype(np.uint8)

        end_time = time.time()

        print('-------------------------')
        print(f'CIFAR-10 loaded successfully in {end_time - start_time: .2f}s')
        print(f'Size: images-> {self.images.shape} \t labels -> {self.labels.shape}')
        print(f'Unique Labels: {np.unique(self.labels)}')
        print(f'Label Names: {self.label_names}')
        print(f'Mean: {self.mean} Std: {self.std}')
        print("-------------------------\n")

        # Normalize according to normalization argument
        if self.normalization == 'z-score':
            self.images = (self.images - self.mean) / self.std
        elif self.normalization == 'min-max':
            self.min = np.min(self.images)
            self.max = np.max(self.images)
            self.images = (self.images - self.min) / (self.max - self.min)

    def statistical_analysis(self):
        """
        Analyze the dataset and extract distribution and example plots
        """

        # 1. Class Distribution Plot
        _, class_counts = np.unique(self.labels, return_counts=True)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=self.label_names, y=class_counts, palette='viridis')
        plt.title('Class Distribution')
        plt.xlabel('Hair Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'class_distribution.png'))
        plt.close()

        # 2. Pixel Value Distribution after Normalization (with downsampling)
        # Downsample by selecting a random subset of pixel values (e.g., 100,000)
        flattened_pixels = self.images.flatten()
        sample_size = min(len(flattened_pixels), 100000)  # Adjust sample size as needed
        sampled_pixels = np.random.choice(flattened_pixels, size=sample_size, replace=False)

        plt.figure(figsize=(8, 6))
        sns.histplot(sampled_pixels, bins=50, kde=True, color='green')
        plt.title('Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'pixel_distribution_z-score.png'))
        plt.close()

        # 3. Display a Random Grid of Images
        random_indices = np.random.choice(len(self.images), size=16, replace=False)
        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(random_indices):
            plt.subplot(4, 4, i + 1)
            # Visualize image with implemented function
            temp_img = self.visualize(idx)
            # Show it in correct position of grid
            plt.imshow(temp_img.astype(np.uint8), cmap='gray')
            plt.title(self.label_names[self.labels[idx]])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'image_grid.png'))
        plt.close()

        print(f'Statistical analysis completed. Plots saved to {self.data_folder}')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


    def visualize(self, idx):
        """
        Turn the cifar-10 data format into a visualized image.
        :param idx: Index to be visualized. (int)
        :return: An RGB image from the cifar-10 dataset as a HWC 32x32 numpy array.
        """
        # Retrieve data from index
        image = self.images[idx]

        # Un-normalize
        if self.normalization == 'z-score':
            image = image * self.std + self.mean
        elif self.normalization == 'min-max':
            image = image * (self.max - self.min) + self.min

        return image



if __name__ == "__main__":
    cifar_dataset = Cifar10()
    cifar_dataset.statistical_analysis()

