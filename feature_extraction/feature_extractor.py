import abc
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FeatureExtractor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract_features(self):
        """ This method needs to be implemented in all the subclasses """

    def calculate_all_roots(self, f, range):
        roots = list()
        sign = 'minus' if f(range[0]) < 0 else 'plus'
        index = 0
        for i in range:
            if sign == 'minus' and f(i) >= 0:
                sign = 'plus'
                try:
                    root = optimize.newton(f, i)
                except ValueError:
                    root = optimize.bisect(f, i, range[index-1])
                roots.append(root)
            if sign == 'plus' and f(i) <= 0:
                sign = 'minus'
                try:
                    root = optimize.newton(f, i)
                except ValueError:
                    root = optimize.bisect(f, i, range[index-1])
                roots.append(root)
            index = index+1
        return roots
