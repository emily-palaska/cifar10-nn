import json
from baselines.cifar10 import Cifar10
from baselines.mlp_hinge_loss import *

def main():
    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    # n_classes = dataset.get_number_of_classes()

    # Train
    num_classes = 10
    hidden_dim = 128
    epochs = 100
    learning_rate = 0.01
    classifiers, results = train_one_vs_one(x_train, y_train, input_dim=3072, hidden_dim=hidden_dim, num_classes=num_classes,
                                   epochs=epochs, learning_rate=learning_rate)

    # Predict
    evaluation_metrics = predict_one_vs_one(classifiers, x_test, y_test)
    results['test'] = evaluation_metrics

    # Save the updated dictionary back to the file
    file_path = f'../results/mlp/mlp_lr{learning_rate}_epochs{epochs}.json'
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()