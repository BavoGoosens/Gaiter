from scipy.sparse import csr_matrix
import numpy as np
from itertools import cycle
import matplotlib
import random
from collections import Counter
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WalkingClassifier(object):
    def __init__(self, data_set, labels):
        self.data_set = data_set
        self.data_set_labels = labels
        self.cluster_labels = list()
        self.unique = list()

    def get_data_set(self):
        return self.data_set

    def make_plot(self, nb_clusters, labels, centers, filename):
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(nb_clusters), colors):
            my_members = labels == k
            cluster_center = centers[k]
            plt.plot(self.data_set[my_members, 0], self.data_set[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.savefig(filename)

    def get_walking_frames(self, cutoff=0.8, method='first', shuffle=False):
        walking_data_set = list()
        walking_data_labels = list()
        count = Counter(self.cluster_labels)
        select = dict()
        if method == 'even':
            for l, c in count.iteritems():
                select[l] = int(np.round(c * cutoff))
        else:
            nb = int(np.round(np.sum(count.values()) * cutoff))
            for l, c in count.iteritems():
                if c < nb and not nb == 0:
                    select[l] = c
                    nb -= c
                elif c >= nb:
                    select[l] = nb
                    nb = 0

        for l, c in select.iteritems():
            indices = [i for i, x in enumerate(self.cluster_labels) if x == l]
            random.shuffle(indices)
            indices[:c]
            for idx in indices:
                walking_data_set.append(self.data_set[idx,:])
                walking_data_labels.append(self.data_set_labels[idx, :])



        return csr_matrix(walking_data_set), csr_matrix(walking_data_labels)