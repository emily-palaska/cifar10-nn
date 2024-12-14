from baselines.cifar10 import Cifar10
import svm.scratch as mysvm

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()

    # One vs all logic
    target = 1
    y_train, y_test = mysvm.one_vs_all(y_train, target), mysvm.one_vs_all(y_test, target)

    # Initialize classifier
    svm_classifier = mysvm.SVMClassifier(learning_rate=0.0001, lambda_param=0.01, n_iters=100)
    svm_classifier.fit(x_train, y_train)
    svm_classifier.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()