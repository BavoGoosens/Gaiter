from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter
from walking_classifier import WalkingClassifier


class KMeansMiniBatch(WalkingClassifier):
    def __init__(self, featured_frames):
        super(KMeansMiniBatch, self).__init__(featured_frames)
        self.labels = list()


    def train(self, nb_clusters):
        data_set = self.data_set.toarray()

        mbk = MiniBatchKMeans(n_clusters=nb_clusters)
        mbk.fit(data_set)
        self.labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(self.labels)

        print("###########################MBK means######################################")
        print(Counter(self.labels))

        self.make_plot(nb_clusters, self.labels, mbk_means_cluster_centers, "img/kmeans_mini_batch.png")