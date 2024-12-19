from sklearn.svm import SVC
from .utils import *

class SVMSklearn:
    def __init__(self, kernel='linear'):
        """
        CIFAR-10 classifier class using Scikit-learn models.
        """
        self.model = None
        self.pca = None
        self.kernel = kernel
        self.model = SVC(kernel=self.kernel, random_state=42)

    def fit(self, x, y):
        """
        Trains the classifier on the given training data.
        """
        if x.ndim != 2:
            x = x.reshape((x.shape[0], -1))

        self.model.fit(x, y)

    def evaluate(self, x_test, y_test):
        """
        Evaluates the classifier on the test data and returns the accuracy.
        """
        if x_test.ndim != 2:
            x_test = x_test.reshape((x_test.shape[0], -1))

        y_pred = self.model.predict(x_test)
        return metrics(y_pred, y_test)

    def predict(self, x):
        """
        Predicts the class for the given input data.
        """
        return self.model.predict(x)
