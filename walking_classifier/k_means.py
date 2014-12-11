from sklearn.cluster import KMeans as KM
import numpy as np
from collections import Counter
from walking_classifier import WalkingClassifier


class KMeans(WalkingClassifier):
    def __init__(self, featured_frames):
        super(KMeans, self).__init__(featured_frames)

    def train(self, nb_clusters):
        data_set = self.data_set.toarray()
        kmeans = KM(n_clusters=nb_clusters, init='k-means++')
        kmeans.fit(data_set)

        self.labels = kmeans.labels_
        k_means_cluster_centers = kmeans.cluster_centers_
        k_means_labels_unique = np.unique(self.labels)

        print("#############################K means######################################")
        print(Counter(self.labels))

        self.make_plot(nb_clusters, self.labels, k_means_cluster_centers, 'img/kmeans.png')


