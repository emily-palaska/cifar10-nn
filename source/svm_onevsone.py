from baselines.cifar10 import Cifar10
import svm.scratch as mysvm

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    n_classes = dataset.get_number_of_classes()

    """
    # Replace cifar10 with noise
    n_classes = 10

    # Number of training and test images
    num_train_images = 1000
    num_test_images = 100

    # Image dimensions
    height, width, channels = 32, 32, 3

    # Generate random noise for training and test datasets
    x_train = np.random.rand(num_train_images, height, width, channels)
    x_test = np.random.rand(num_test_images, height, width, channels)

    # Labels can also be random integers between 0 and 9 (10 classes for CIFAR-10)
    y_train = np.random.randint(0, 10, size=(num_train_images,))
    y_test = np.random.randint(0, 10, size=(num_test_images,))
    """

    # One vs all strategy
    learning_rate = 0.0001
    lambda_param = 0.01
    n_iters = 10
    kernel = 'rbf'
    degree = 3
    file_name = f'../results/svm/1v1_lr{learning_rate}_l{lambda_param}_n{n_iters}_{kernel}{degree}.json'
    classifier = mysvm.OneVsOneClassifier(learning_rate=learning_rate,
                                          lambda_param=lambda_param,
                                          n_iters=n_iters,
                                          results_file=file_name,
                                          kernel=kernel,
                                          degree=degree
                                          )
    classifier.fit(x_train, y_train, n_classes)
    evaluation = classifier.evaluate(x_test, y_test, n_classes)

    print("Evaluation Metrics:", evaluation)

if __name__ == '__main__':
    main()