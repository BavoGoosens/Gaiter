import numpy as np
from sklearn.cluster import DBSCAN as DB
from collections import Counter
from walking_classifier import WalkingClassifier
import matplotlib
from collections import Counter

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DBScan(WalkingClassifier):
    def __init__(self, featured_frame):
        super(DBScan, self).__init__(featured_frame)

    def train(self):
        data_set = self.data_set.toarray()
        db = DB().fit(data_set)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.labels = db.labels_

        print("############################ DBScan  ######################################")
        print(Counter(self.labels))
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        # Black removed and is used for noise instead.
        unique_labels = set(self.labels)
        self.unique = unique_labels
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (self.labels == k)

            xy = data_set[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = data_set[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('img/dbscan.png')



