import numpy as np
from sklearn.cluster import MeanShift as MS, estimate_bandwidth
from collections import Counter
from walking_classifier import WalkingClassifier


class MeanShift(WalkingClassifier):
    def __init__(self, data_set, labels):
        super(MeanShift, self).__init__(data_set, labels)

    def train(self):
        # Compute clustering with MeanShift

        # The following bandwidth can be automatically detected using
        data_set = self.data_set
        bandwidth = estimate_bandwidth(data_set, quantile=0.2, n_samples=500)

        ms = MS(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data_set)
        self.cluster_labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(self.cluster_labels)

        self.set_classifier(ms)
        self.unique = labels_unique
        n_clusters_ = len(labels_unique)

        print("###########################Mean Shift######################################")
        print(Counter(self.cluster_labels))
        print("number of estimated clusters : %d" % n_clusters_)

        self.make_plot(n_clusters_, self.cluster_labels, cluster_centers, "img/mean_shift_test.png")