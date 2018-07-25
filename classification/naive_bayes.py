from utils.activations import softmax
from base import BaseEstimator
import numpy as np

class NaiveBayesClassifier(BaseEstimator):

    def __init__(self, n_classes=2):
        self.n_classes = n_classes


    def fit(self, X, y=None):
        self._setup_input(X, y)
        assert list(np.unique(y)) == list(range(self.n_classes))

        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon = 1e-5 * np.var(X, axis=0).max()

        for c in range(self.n_classes):
            X_c = X[y == c]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0) + self.epsilon
            self._priors[c] = X_c.shape[0] / X.shape[0]


    def _predict(self, X=None):
        predictions = np.apply_along_axis(self._predict_row, 1, X)
        return softmax(predictions)


    def _predict_row(self, x):
        output = []
        for c in range(self.n_classes):
            # prior = np.log(self._priors[c])
            # likelihood = np.log(self._pdf(c, x)).sum()
            # posterior = prior + likelihood

            prior = self._priors[c]
            likelihood = self._pdf(c, x).prod()
            posterior = prior * likelihood

            output.append(posterior)

        return output


    def _pdf(self, n_class, x):
        mean = self._mean[n_class]
        var = self._var[n_class]
        num = np.exp(- (x - mean)**2 / (2*var))
        den = np.sqrt(2 * np.pi * var)
        return num / den

