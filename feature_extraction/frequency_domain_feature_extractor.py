from feature_extractor import *
from data_utils.featured_frame import *

import scipy.fftpack as ff
import numpy as np

# TODO
class FrequencyDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)
        self.calculate_mean(frame)
        return frame


    def calculate_linear_coefficients(self):
        pass

    def calculate_mel_scale_coeficients(self):
        pass

    def calculate_spectral_coefficients(self):
        pass

    def calculate_spectral_energy(self):
        pass

    def calculate_spectral_entropy(self):
        pass

    def calculate_ac_component(self):
        pass

    def calculate_dc_component(self):
        pass

    def calculate_mean(self, frame):
        x_axis = ff.fft(frame.get_x_data())
        y_axis = ff.fft(frame.get_y_data())
        z_axis = ff.fft(frame.get_z_data())
        frame.add_feature('x_mean_fd', np.mean(x_axis))
        frame.add_feature('y_mean_fd', np.mean(y_axis))
        frame.add_feature('z_mean_fd', np.mean(z_axis))

    def calculate_correlation(self):
        pass

    def calculate_spectral_roll_off(self):
        pass

    def calculate_spectral_centeroid(self):
        pass

    def calculate_spectral_flux(self):
        pass