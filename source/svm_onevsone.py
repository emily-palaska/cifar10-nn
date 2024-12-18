from baselines.cifar10 import Cifar10
import svm.scratch as mysvm

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    n_classes = dataset.get_number_of_classes()

    # One vs all strategy
    learning_rate = 0.0001
    lambda_param = 0.01
    n_iters = 10
    kernel = 'polynomial'
    degree = 3
    file_name = f'../results/svm/1v1_lr{learning_rate}_l{lambda_param}_n{n_iters}_k{kernel}{degree}.json'
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