from personal_classifier import PersonalClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

class SupportVectorMachine(PersonalClassifier):
    def __init__(self, featured_frames):
        super(SupportVectorMachine, self).__init__(featured_frames)

    def train(self, kernel='linear'):
        X = self.data_set.toarray()
        y = np.array(self.labels)

        n_sample = len(X)

        np.random.seed(0)
        order = np.random.permutation(n_sample)
        X = X[order]
        y = y[order].astype(np.float)

        X_train = X[:.9 * n_sample]
        y_train = y[:.9 * n_sample]
        X_test = X[.9 * n_sample:]
        y_test = y[.9 * n_sample:]

        clf = svm.SVC(kernel=kernel, gamma=10)
        clf.fit(X_train, y_train)