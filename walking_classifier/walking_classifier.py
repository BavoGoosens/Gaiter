from scipy.sparse import csr_matrix
import numpy as np
from itertools import cycle
import matplotlib
from collections import Counter

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WalkingClassifier(object):
    def __init__(self, featured_frames):
        self.featured_frames = featured_frames
        self.data_set = csr_matrix(self.flatten(featured_frames))
        self.labels = list()

    def flatten(self, featured_frame_list):
        flat_list = list()
        for f_frame in featured_frame_list:
            features = f_frame.get_flat_features()
            flat_list.append(features)
        return flat_list

    def get_data_set(self):
        return self.data_set

    def make_plot(self, nb_clusters, labels, centers, filename):
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(nb_clusters), colors):
            my_members = labels == k
            cluster_center = centers[k]
            plt.plot(self.data_set.toarray()[my_members, 0], self.data_set.toarray()[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.savefig(filename)

    def save_sparse_csr(self, filename, array):
        np.savez(filename, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

    def get_walking_frames(self, cutoff=0.8, method='even'):
        walking_data = list()
        if method == 'even':
            count = Counter(self.labels)
            count = count.values()
            select = [np.round(c * cutoff) for c in count]
            rel_count = None
        else:
            pass