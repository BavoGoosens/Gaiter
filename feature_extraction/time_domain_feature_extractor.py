from data_utils import featured_frame
from data_utils.featured_frame import FeaturedFrame
from feature_extractor import *
from data_utils.featured_frame import *

import numpy as np
import scipy.stats as sts
import scipy.integrate as int

class TimeDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)
        self.add_mean(frame)
        self.add_integral(frame)
        self.add_kurtosis(frame)
        self.add_abs_mean(frame)
        self.add_variance(frame)
        self.add_minimum(frame)
        self.add_maximum(frame)
        self.add_abs_minimum(frame)
        self.add_abs_maximum(frame)
        self.add_root_mean_square(frame)
        self.add_min_max_difference(frame)
        self.add_percentiles(frame)
        self.add_skewness(frame)
        self.add_median(frame)
        self.add_std(frame)
        return frame

    def add_mean(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_mean', np.mean(x_axis))
        frame.add_feature('y_mean', np.mean(y_axis))
        frame.add_feature('z_mean', np.mean(z_axis))

    def add_integral(self, frame):
        t_axis = frame.get_t_data()
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_integral', int.simps(x_axis))
        frame.add_feature('y_integral', int.simps(y_axis))
        frame.add_feature('z_integral', int.simps(z_axis))

    def add_kurtosis(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_kurtosis', sts.kurtosis(x_axis))
        frame.add_feature('y_kurtosis', sts.kurtosis(y_axis))
        frame.add_feature('z_kurtosis', sts.kurtosis(z_axis))

    def add_abs_mean(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_abs_mean', abs(np.mean(x_axis)))
        frame.add_feature('y_abs_mean', abs(np.mean(y_axis)))
        frame.add_feature('z_abs_mean', abs(np.mean(z_axis)))

    def add_variance(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_variance', np.var(x_axis))
        frame.add_feature('y_variance', np.var(y_axis))
        frame.add_feature('z_variance', np.var(z_axis))

    def add_minimum(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_min', min(x_axis))
        frame.add_feature('y_min', min(y_axis))
        frame.add_feature('z_min', min(z_axis))

    def add_maximum(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_max', max(x_axis))
        frame.add_feature('y_max', max(y_axis))
        frame.add_feature('z_max', max(z_axis))

    def add_abs_minimum(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_abs_min', abs(min(x_axis)))
        frame.add_feature('y_abs_min', abs(min(y_axis)))
        frame.add_feature('z_abs_min', abs(min(z_axis)))

    def add_abs_maximum(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_abs_max', abs(max(x_axis)))
        frame.add_feature('y_abs_max', abs(max(y_axis)))
        frame.add_feature('z_abs_max', abs(max(z_axis)))

    def add_root_mean_square(self, frame):
        x_axis_pow = np.power(frame.get_x_data(), 2)
        y_axis_pow = np.power(frame.get_y_data(), 2)
        z_axis_pow = np.power(frame.get_z_data(), 2)
        frame.add_feature('x_rms', np.sqrt(np.mean(x_axis_pow)))
        frame.add_feature('y_rms', np.sqrt(np.mean(y_axis_pow)))
        frame.add_feature('z_rms', np.sqrt(np.mean(z_axis_pow)))

    def add_min_max_difference(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_min_max_difference', max(x_axis) - min(x_axis))
        frame.add_feature('y_min_max_difference', max(y_axis) - min(y_axis))
        frame.add_feature('z_min_max_difference', max(z_axis) - min(z_axis))

    def add_percentiles(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_25_percentile', np.percentile(x_axis, 25))
        frame.add_feature('y_25_percentile', np.percentile(y_axis, 25))
        frame.add_feature('z_25_percentile', np.percentile(z_axis, 25))
        frame.add_feature('x_75_percentile', np.percentile(x_axis, 75))
        frame.add_feature('y_75_percentile', np.percentile(y_axis, 75))
        frame.add_feature('z_75_percentile', np.percentile(z_axis, 75))

    def add_skewness(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_skewness', sts.skew(x_axis))
        frame.add_feature('y_skewness', sts.skew(y_axis))
        frame.add_feature('z_skewness', sts.skew(z_axis))

    def add_median(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_median', np.median(x_axis))
        frame.add_feature('y_median', np.median(y_axis))
        frame.add_feature('z_median', np.median(z_axis))

    def add_std(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_feature('x_std', np.std(x_axis))
        frame.add_feature('y_std', np.std(y_axis))
        frame.add_feature('z_std', np.std(z_axis))

    def calculate_peak_distances(self, a):
        peaks = self.peakdetect(a)
        max_peaks = peaks[0]
        min_peaks = peaks[1]
        print max_peaks
        print min_peaks
        distances = list()
        k = len(max_peaks)-1 if len(max_peaks) < len(min_peaks) else len(min_peaks)
        for i in range(0,k):
            distances.append(max_peaks[i] - min_peaks[i])
        return distances

    def add_avg_peak_distance(self, frame):
        x_peak_distances = self.calculate_peak_distances(frame.get_x_data())
        y_peak_distances = self.calculate_peak_distances(frame.get_y_data())
        z_peak_distances = self.calculate_peak_distances(frame.get_z_data())
        frame.add_feature('x_avg_peak_dist', np.mean(x_peak_distances))
        frame.add_feature('y_avg_peak_dist', np.mean(y_peak_distances))
        frame.add_feature('z_avg_peak_dist', np.mean(z_peak_distances))

    def calculate_peak_mean(self, frame):
        pass

    def calculate_variance_in_peak(self, frame):
        pass

    def calculate_apf(self, frame):
        pass

    def calculate_variance_apf(self, frame):
        pass

    def calculate_root_mean_square_of_integral(self, frame):
        pass

    def calculate_mean_min_max_sums(self, frame):
        pass
