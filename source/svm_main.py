from svm.scratch.svm_classifier import SVMClassifier
from baselines.cifar10 import Cifar10

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()

    # add one vs all logic or one vs one logic

    svm_classifier = SVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm_classifier.fit(x_train, y_train)
    svm_classifier.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()