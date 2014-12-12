from scipy.sparse import csr_matrix
import numpy as np

class PersonalClassifier(object):
    def __init__(self, data_set, labels):
        self.data_set = data_set
        self.labels = labels

    def set_classifier(self, clf):
        self.classifier = clf

    def get_classifier(self):
        return self.classifier

    def classify(self, featured_frames):
        return self.classifier.predict(featured_frames)
