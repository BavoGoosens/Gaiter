import abc
import numpy as np
from scipy import optimize

class FeatureExtractor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract_features(self):
        """ This method needs to be implemented in all the subclasses """

    def calculate_all_roots(self, f, range):
        roots = list()
        sign = 'minus' if f(1) < 0 else 'plus'
        for i in range:
            if sign == 'minus' and f(i) >= 0:
                sign = 'plus'
                root = optimize.newton(f, i)
                roots.append(root)
            if sign == 'plus' and f(i) <= 0:
                sign = 'minus'
                root = optimize.newton(f, i)
                roots.append(root)
        return roots
