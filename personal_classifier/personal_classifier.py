from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter


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

    def label_data(self, unlabeld_data, selector=None):
        labels = dict()
        for key, value in unlabeld_data.iteritems():
            data_set = self.flatten(value)
            if selector is not None:
                data_set = selector.transform(data_set)
            pred = self.classify(data_set)
            c = Counter(pred)
            labels[key] = c.most_common(1)
        return labels

    def flatten(self ,featured_frame_list):
        flat_list = list()
        for f_frame in featured_frame_list:
            features = f_frame.get_flat_features()
            flat_list.append(features)
        return flat_list