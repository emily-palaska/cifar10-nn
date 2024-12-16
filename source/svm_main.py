from baselines.cifar10 import Cifar10
import svm.scratch as mysvm

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    n_classes = dataset.get_number_of_classes()

    # One vs all strategy
    classifier = mysvm.OneVsAllClassifier(learning_rate=0.0001, lambda_param=0.01, n_iters=100, results_file='../results/svm/one_vs_all_results.json')
    classifier.fit(x_train, y_train, n_classes)
    evaluation = classifier.evaluate(x_test, y_test)

    print("Evaluation Metrics:", evaluation)

if __name__ == '__main__':
    main()