from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter
from walking_classifier import WalkingClassifier


class KMeansMiniBatch(WalkingClassifier):
    def __init__(self, data_set, labels):
        super(KMeansMiniBatch, self).__init__(data_set, labels)
        self.cluster_labels = list()

    def train(self, nb_clusters):
        data_set = self.data_set

        mbk = MiniBatchKMeans(n_clusters=nb_clusters)
        mbk.fit(data_set)
        self.cluster_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(self.cluster_labels)

        self.set_classifier(mbk)
        self.unique = mbk_means_labels_unique
        print("###########################MBK means######################################")
        print(Counter(self.cluster_labels))

        self.make_plot(nb_clusters, self.cluster_labels, mbk_means_cluster_centers, "img/kmeans_mini_batch.png")