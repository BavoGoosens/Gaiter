from data_utils import featured_frame
from data_utils.featured_frame import FeaturedFrame
from feature_extractor import *
from data_utils.featured_frame import *

import numpy as np

class TimeDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        frame = FeaturedFrame(frame)
        self.calculate_mean(frame)
        self.calculate_integral(frame)
        self.calculate_kurtosis(frame)
        self.calculate_abs_mean(frame)
        self.calculate_variance(frame)
        self.calculate_minimum(frame)
        self.calculate_maximum(frame)
        self.calculate_abs_minimum(frame)
        self.calculate_abs_maximum(frame)
        self.calculate_avg_peak_distance(frame)
        self.calculate_peak_mean(frame)
        self.calculate_variance_in_peak(frame)
        self.calculate_apf(frame)
        self.calculate_variance_apf(frame)
        self.calculate_root_mean_square(frame)
        self.calculate_min_max_difference(frame)
        self.calculate_percentiles(frame)
        self.calculate_skewness(frame)
        self.calculate_median(frame)
        self.calculate_root_mean_square_of_integral(frame)
        self.calculate_mean_min_max_sums(frame)
        return frame

    def calculate_mean(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_mean', np.mean(x_axis))
        frame.add_feature('y_mean', np.mean(y_axis))
        frame.add_feature('z_mean', np.mean(z_axis))

    def calculate_integral(self, frame):
        pass

    def calculate_kurtosis(self, frame):
        pass

    def calculate_abs_mean(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_abs_mean', abs(np.mean(x_axis)))
        frame.add_feature('y_abs_mean', abs(np.mean(y_axis)))
        frame.add_feature('z_abs_mean', abs(np.mean(z_axis)))

    def calculate_variance(self, frame):
        pass

    def calculate_minimum(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_min', min(x_axis))
        frame.add_feature('y_min', min(y_axis))
        frame.add_feature('z_min', min(z_axis))

    def calculate_maximum(self, frame):
        pass

    def calculate_abs_minimum(self, frame):
        pass

    def calculate_abs_maximum(self, frame):
        pass

    def calculate_avg_peak_distance(self, frame):
        pass

    def calculate_peak_mean(self, frame):
        pass

    def calculate_variance_in_peak(self, frame):
        pass

    def calculate_apf(self, frame):
        pass

    def calculate_variance_apf(self, frame):
        pass

    def calculate_root_mean_square(self, frame):
        pass

    def calculate_min_max_difference(self, frame):
        pass

    def calculate_percentiles(self, frame):
        pass

    def calculate_skewness(self, frame):
        pass

    def calculate_median(self, frame):
        pass

    def calculate_root_mean_square_of_integral(self, frame):
        pass

    def calculate_mean_min_max_sums(self, frame):
        pass