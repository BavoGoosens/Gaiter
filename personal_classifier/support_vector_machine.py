from personal_classifier import PersonalClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm


class SupportVectorMachine(PersonalClassifier):
    def __init__(self, data_set, labels):
        super(SupportVectorMachine, self).__init__(data_set, labels)

    def train(self, kernel='linear'):
        x = self.data_set
        y = self.labels

        clf = svm.SVC(kernel=kernel, gamma=10)
        clf.fit(x, y)

        self.set_classifier(clf)