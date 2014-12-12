from sklearn.cluster import KMeans as KM
import numpy as np
from collections import Counter
from walking_classifier import WalkingClassifier


class KMeans(WalkingClassifier):
    def __init__(self, data_set, labels):
        super(KMeans, self).__init__(data_set, labels)

    def train(self, nb_clusters):
        data_set = self.data_set
        kmeans = KM(n_clusters=nb_clusters, init='k-means++')
        kmeans.fit(data_set)

        self.cluster_labels = kmeans.labels_
        k_means_cluster_centers = kmeans.cluster_centers_
        k_means_labels_unique = np.unique(self.cluster_labels)
        self.unique = k_means_labels_unique
        print("#############################K means######################################")
        print(Counter(self.cluster_labels))

        self.make_plot(nb_clusters, self.cluster_labels, k_means_cluster_centers, 'img/kmeans.png')


