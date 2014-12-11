from frame import *

from scipy.interpolate import interp1d

import numpy as np



class FeaturedFrame(Frame):
    def __init__(self, frame):
        super(FeaturedFrame, self).__init__(frame.get_frame_data(), frame.get_size(), frame.get_overlap(),
                                            frame.get_raw_data())
        self.features = dict()
        self.derivatives = dict()
        self.coefficients = dict()
        self.peaks = dict()


    def add_feature(self, name, value):
        self.features[name] = value

    def get_feature(self, name):
        return self.features[name]

    def add_coefficients(self, name, value):
        self.coefficients[name] = value

    def get_coefficients(self, name):
        return self.coefficients[name]

    def add_derivative(self, name, value):
        self.derivatives[name] = value

    def get_derivative(self, name):
        return self.derivatives[name]

    def add_peaks(self, name, value):
        self.peaks[name] = value

    def get_peaks(self, name):
        return self.peaks[name]

    def get_function(self, type):
        if type == 'x':
            x = self.get_x_data()
            t = np.linspace(0, len(x)-1, len(x))
            f = interp1d(t, x)
        elif type == 'y':
            y = self.get_y_data()
            t = np.linspace(0, len(y)-1, len(y))
            f = interp1d(t, y)
        elif type == 'z':
            z = self.get_z_data()
            t = np.linspace(0, len(z)-1, len(z))
            f = interp1d(t, z)
        elif type == 'x_der':
            x = self.get_derivative('x')
            t = np.linspace(1, len(x)-2, len(x))
            f = interp1d(t, x)
        elif type == 'y_der':
            y = self.get_derivative('y')
            t = np.linspace(1, len(y)-2, len(y))
            f = interp1d(t, y)
        elif type == 'z_der':
            z = self.get_derivative('z')
            t = np.linspace(1, len(z)-2, len(z))
            f = interp1d(t, z)
        elif type == 'x_der2':
            x = self.get_derivative('x2')
            t = np.linspace(1, len(x)-2, len(x))
            f = interp1d(t, x)
        elif type == 'y_der2':
            y = self.get_derivative('y2')
            t = np.linspace(1, len(y)-2, len(y))
            f = interp1d(t, y)
        elif type == 'z_der2':
            z = self.get_derivative('z2')
            t = np.linspace(1, len(z)-2, len(z))
            f = interp1d(t, z)
        else:
            raise AttributeError("Wrong attribute: "+str(type))
        return f


    def get_all_features(self):
        return self.features

    def get_flat_features(self):
        feats = list()
        for value in self.features.itervalues():
            feats.append(value)
        for value in self.coefficients.itervalues():
            # nb of coeffs to use
            coeffs = value[:5]
            for c in coeffs:
                feats.append(c)
        return feats
