import abc
import numpy as np
from scipy import optimize, stats
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_all_roots(f, rng):
    roots = list()
    sign = 'minus' if f(rng[0]) < 0 else 'plus'
    index = 0
    for i in rng:
        if sign == 'minus' and f(i) >= 0:
            sign = 'plus'
            try:
                root = optimize.newton(f, i)
            except ValueError:
                root = optimize.bisect(f, i, rng[index - 1])
            roots.append(root)
        if sign == 'plus' and f(i) <= 0:
            sign = 'minus'
            try:
                root = optimize.newton(f, i)
            except ValueError:
                root = optimize.bisect(f, i, rng[index - 1])
            roots.append(root)
        index += 1
    return roots


def add_pearson(frame, derivative=True, domain='time'):
    der = ''
    if domain == 'spec':
        if derivative:
            der = '_der'
        x_axis = frame.get_coefficients('x_spectral_cos' + der)
        y_axis = frame.get_coefficients('y_spectral_cos' + der)
        z_axis = frame.get_coefficients('z_spectral_cos' + der)
        frame.add_feature('x_y_pearson' + der + "_" + domain, stats.pearsonr(x_axis, y_axis)[0])
        frame.add_feature('y_z_pearson' + der + "_" + domain, stats.pearsonr(y_axis, z_axis)[0])
        frame.add_feature('x_z_pearson' + der + "_" + domain, stats.pearsonr(x_axis, z_axis)[0])
    else:
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_y_pearson' + der + "_" + domain, stats.pearsonr(x_axis, y_axis)[0])
        frame.add_feature('y_z_pearson' + der + "_" + domain, stats.pearsonr(y_axis, z_axis)[0])
        frame.add_feature('x_z_pearson' + der + "_" + domain, stats.pearsonr(x_axis, z_axis)[0])
    return frame


class FeatureExtractor(object):
    def __init__(self, derivative):
        self.derivative = derivative

    @abc.abstractmethod
    def extract_features(self):
        """ This method needs to be implemented in all the subclasses """
