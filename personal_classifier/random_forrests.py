from personal_classifier import PersonalClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RandomForrest(PersonalClassifier):
    def __init__(self, data_set, labels):
        super(RandomForrest, self).__init__(data_set, labels)

    def train(self):
        x = self.data_set
        y = self.labels

        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(x, y)

        self.set_classifier(clf)