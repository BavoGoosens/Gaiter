from feature_extractor import FeatureExtractor
from data_utils.featured_frame import *
import pywt


class WaveletFeatureExtractor(FeatureExtractor):
    def __init__(self, derivative=True):
        super(WaveletFeatureExtractor, self).__init__(derivative)

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)
        self.add_decomposition_coefficients(frame)
        return frame

    def add_decomposition_coefficients(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_coefficients('x_wavelet_cos', self.calculate_decomposition_coefficients(x_axis))
        frame.add_coefficients('y_wavelet_cos', self.calculate_decomposition_coefficients(y_axis))
        frame.add_coefficients('z_wavelet_cos', self.calculate_decomposition_coefficients(z_axis))

    def calculate_decomposition_coefficients(self, data):
        cA, cD = pywt.dwt(data, 'db1')
        return cD