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
    def __init__(self, featured_frames):
        self.featured_frames = featured_frames
        self.data_set = csr_matrix(self.flatten(featured_frames))
        self.labels = list()
        self.unique = list()

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

    def get_walking_frames(self, cutoff=0.8, method='first', shuffle=False):
        walking_data = list()
        clusters = defaultdict(list)
        for idx, label in enumerate(self.labels):
            clusters[label].append(self.featured_frames[idx])
        if shuffle:
            for l in clusters.itervalues():
                random.shuffle(l)
        count = Counter(self.labels)
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
        for label, number in select.iteritems():
            data = clusters[label][:number]
            walking_data.extend(data)
        return walking_data