import os

from sklearn.svm import LinearSVC, SVC
from .utils import *

class SVMSklearn:
    def __init__(self, kernel='linear'):
        """
        CIFAR-10 classifier class using Scikit-learn models.
        """
        self.model = None
        self.pca = None
        self.kernel = kernel
        self.model = SVC(random_state=42, verbose=True, kernel=self.kernel)

    def fit(self, x, y):
        """
        Trains the classifier on the given training data.
        """
        if x.ndim != 2:
            x = x.reshape((x.shape[0], -1))

        self.model.fit(x, y)

    def evaluate(self, x_test, y_test, file_name=None, save_dir='../results/svm/'):
        """
        Evaluates the classifier on the test data and returns the accuracy.
        """
        if x_test.ndim != 2:
            x_test = x_test.reshape((x_test.shape[0], -1))

        if file_name is None:
            file_name = f'sklearn_{self.kernel}.json'

        y_pred = self.model.predict(x_test)
        evaluation_metrics = metrics(y_pred, y_test)

        with open(os.path.join(save_dir, file_name), "w") as f:
            json.dump(evaluation_metrics, f, indent=4)


    def predict(self, x):
        """
        Predicts the class for the given input data.
        """
        return self.model.predict(x)
