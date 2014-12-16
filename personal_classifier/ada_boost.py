from sklearn.ensemble import AdaBoostClassifier
from personal_classifier import PersonalClassifier


class AdaBoost(PersonalClassifier):
    def __init__(self, data_set, labels):
        super(AdaBoost, self).__init__(data_set, labels)

    def train(self):
        x = self.data_set
        y = self.labels

        clf = AdaBoostClassifier()
        clf.fit(x, y)

        self.set_classifier(clf)