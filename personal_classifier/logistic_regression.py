from personal_classifier import PersonalClassifier
from sklearn.linear_model import LogisticRegression as LR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression(PersonalClassifier):
    def __init__(self, data_set, labels):
        super(LogisticRegression, self).__init__(data_set, labels)

    def train(self):
        x = self.data_set
        y = self.labels

        log_reg = LR(C=1e5)

        # we create an instance of Neighbours Classifier and fit the data.
        log_reg.fit(x, y)

        self.set_classifier(log_reg)
