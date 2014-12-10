from feature_extractor import *
from data_utils.featured_frame import *

import scipy.fftpack as ff
import numpy as np

class FrequencyDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)
        self.add_spectral_coefficients(frame)
        self.add_spectral_mean(frame)
        self.add_spectral_energy(frame)
        self.add_dc_component(frame)
        return frame

    def add_spectral_coefficients(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_spectral_cos', self.calculate_spectral_coefficients(x_axis))
        frame.add_feature('y_spectral_cos', self.calculate_spectral_coefficients(y_axis))
        frame.add_feature('z_spectral_cos', self.calculate_spectral_coefficients(z_axis))

    def calculate_spectral_coefficients(self, data):
        return ff.fft(data)

    def add_mel_scale_coefficients(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_MFCC', self.calculate_mel_scale_coefficients(x_axis))
        frame.add_feature('y_MFCC', self.calculate_mel_scale_coefficients(y_axis))
        frame.add_feature('z_MFCC', self.calculate_mel_scale_coefficients(z_axis))

    def add_spectral_mean(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_spectral_mean', np.mean(self.calculate_spectral_coefficients(x_axis)))
        frame.add_feature('y_spectral_mean', np.mean(self.calculate_spectral_coefficients(y_axis)))
        frame.add_feature('z_spectral_mean', np.mean(self.calculate_spectral_coefficients(z_axis)))

    def add_spectral_energy(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_spectral_energy', np.mean(np.power(self.calculate_spectral_coefficients(x_axis), 2)))
        frame.add_feature('y_spectral_energy', np.mean(np.power(self.calculate_spectral_coefficients(y_axis), 2)))
        frame.add_feature('z_spectral_energy', np.mean(np.power(self.calculate_spectral_coefficients(z_axis), 2)))

    def add_dc_component(self, frame):
        x_axis, y_axis, z_axis = frame.get_x_data(), frame.get_y_data(), frame.get_z_data()
        '''x_spectral, y_spectral, z_spectral = self.calculate_spectral_coefficients(x_axis), self.calculate_spectral_coefficients(y_axis), self.calculate_spectral_coefficients(z_axis)
        x_dc = 0 if len(x_spectral) == 0 else x_spectral[0]
        y_dc = 0 if len(y_spectral) == 0 else y_spectral[0]
        z_dc = 0 if len(z_spectral) == 0 else z_spectral[0]'''
        x_dc = np.mean(x_axis)
        y_dc = np.mean(y_axis)
        z_dc = np.mean(z_axis)
        frame.add_feature('x_dc', x_dc)
        frame.add_feature('y_dc', y_dc)
        frame.add_feature('z_dc', z_dc)

    def add_linear_coefficients(self):
        pass







    def calculate_spectral_entropy(self):
        pass

    def calculate_ac_component(self):
        pass

    def calculate_correlation(self):
        pass

    def calculate_spectral_roll_off(self):
        pass

    def calculate_spectral_centeroid(self):
        pass

    def calculate_spectral_flux(self):
        pass