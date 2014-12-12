from data_utils import featured_frame
from data_utils.featured_frame import FeaturedFrame
from feature_extractor import *
from data_utils.featured_frame import *

import numpy as np
import scipy.stats as sts
import scipy.integrate as integr

from scipy.interpolate import interp1d
from sympy import mpmath

class TimeDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)

        # add derivatives
        self.add_derivative_coefficients(frame)
        self.add_second_derivative_coefficients(frame)

        # add peaks
        self.add_peaks(frame)

        # add features
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
        self.add_avg_min_peak_distance(frame)
        self.add_avg_max_peak_distance(frame)
        self.add_mean_min_peaks(frame)
        self.add_mean_max_peaks(frame)
        self.add_variance_min_peaks(frame)
        self.add_variance_max_peaks(frame)

        # add derivative features
        self.add_mean(frame, True)
        self.add_integral(frame, True)
        self.add_kurtosis(frame, True)
        self.add_abs_mean(frame, True)
        self.add_variance(frame, True)
        self.add_minimum(frame, True)
        self.add_maximum(frame, True)
        self.add_abs_minimum(frame, True)
        self.add_abs_maximum(frame, True)
        self.add_root_mean_square(frame, True)
        self.add_min_max_difference(frame, True)
        self.add_percentiles(frame, True)
        self.add_skewness(frame, True)
        self.add_median(frame, True)
        self.add_std(frame, True)

        return frame


    # ADD FEATURES

    def add_mean(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_mean'+der, np.mean(x_axis))
        frame.add_feature('y_mean'+der, np.mean(y_axis))
        frame.add_feature('z_mean'+der, np.mean(z_axis))

    def add_integral(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_integral'+der, integr.simps(x_axis))
        frame.add_feature('y_integral'+der, integr.simps(y_axis))
        frame.add_feature('z_integral'+der, integr.simps(z_axis))

    def add_kurtosis(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_kurtosis'+der, sts.kurtosis(x_axis))
        frame.add_feature('y_kurtosis'+der, sts.kurtosis(y_axis))
        frame.add_feature('z_kurtosis'+der, sts.kurtosis(z_axis))

    def add_abs_mean(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_abs_mean'+der, abs(np.mean(x_axis)))
        frame.add_feature('y_abs_mean'+der, abs(np.mean(y_axis)))
        frame.add_feature('z_abs_mean'+der, abs(np.mean(z_axis)))

    def add_variance(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_variance'+der, np.var(x_axis))
        frame.add_feature('y_variance'+der, np.var(y_axis))
        frame.add_feature('z_variance'+der, np.var(z_axis))

    def add_minimum(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_min'+der, min(x_axis))
        frame.add_feature('y_min'+der, min(y_axis))
        frame.add_feature('z_min'+der, min(z_axis))

    def add_maximum(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_max'+der, max(x_axis))
        frame.add_feature('y_max'+der, max(y_axis))
        frame.add_feature('z_max'+der, max(z_axis))

    def add_abs_minimum(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_abs_min'+der, abs(min(x_axis)))
        frame.add_feature('y_abs_min'+der, abs(min(y_axis)))
        frame.add_feature('z_abs_min'+der, abs(min(z_axis)))

    def add_abs_maximum(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_abs_max'+der, abs(max(x_axis)))
        frame.add_feature('y_abs_max'+der, abs(max(y_axis)))
        frame.add_feature('z_abs_max'+der, abs(max(z_axis)))

    def add_root_mean_square(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis_pow = np.power(frame.get_derivative('x'), 2)
            y_axis_pow = np.power(frame.get_derivative('y'), 2)
            z_axis_pow = np.power(frame.get_derivative('z'), 2)
            der = '_der'
        else:
            x_axis_pow = np.power(frame.get_x_data(), 2)
            y_axis_pow = np.power(frame.get_y_data(), 2)
            z_axis_pow = np.power(frame.get_z_data(), 2)
        frame.add_feature('x_rms'+der, np.sqrt(np.mean(x_axis_pow)))
        frame.add_feature('y_rms'+der, np.sqrt(np.mean(y_axis_pow)))
        frame.add_feature('z_rms'+der, np.sqrt(np.mean(z_axis_pow)))

    def add_min_max_difference(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_min_max_difference'+der, max(x_axis) - min(x_axis))
        frame.add_feature('y_min_max_difference'+der, max(y_axis) - min(y_axis))
        frame.add_feature('z_min_max_difference'+der, max(z_axis) - min(z_axis))

    def add_percentiles(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_25_percentile'+der, np.percentile(x_axis, 25))
        frame.add_feature('y_25_percentile'+der, np.percentile(y_axis, 25))
        frame.add_feature('z_25_percentile'+der, np.percentile(z_axis, 25))
        frame.add_feature('x_75_percentile'+der, np.percentile(x_axis, 75))
        frame.add_feature('y_75_percentile'+der, np.percentile(y_axis, 75))
        frame.add_feature('z_75_percentile'+der, np.percentile(z_axis, 75))

    def add_skewness(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_skewness'+der, sts.skew(x_axis))
        frame.add_feature('y_skewness'+der, sts.skew(y_axis))
        frame.add_feature('z_skewness'+der, sts.skew(z_axis))

    def add_median(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_median'+der, np.median(x_axis))
        frame.add_feature('y_median'+der, np.median(y_axis))
        frame.add_feature('z_median'+der, np.median(z_axis))

    def add_std(self, frame, derivative = False):
        der = ''
        if derivative:
            x_axis = frame.get_derivative('x')
            y_axis = frame.get_derivative('y')
            z_axis = frame.get_derivative('z')
            der = '_der'
        else:
            x_axis = frame.get_x_data()
            y_axis = frame.get_y_data()
            z_axis = frame.get_z_data()
        frame.add_feature('x_std'+der, np.std(x_axis))
        frame.add_feature('y_std'+der, np.std(y_axis))
        frame.add_feature('z_std'+der, np.std(z_axis))

    def add_avg_min_peak_distance(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_min'), frame.get_peaks('y_min'), frame.get_peaks('z_min')
        frame.add_feature('x_avg_min_peak_distance', self.calculate_avg_distance(x_peaks))
        frame.add_feature('y_avg_min_peak_distance', self.calculate_avg_distance(y_peaks))
        frame.add_feature('z_avg_min_peak_distance', self.calculate_avg_distance(z_peaks))

    def add_avg_max_peak_distance(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_max'), frame.get_peaks('y_max'), frame.get_peaks('z_max')
        frame.add_feature('x_avg_max_peak_distance', self.calculate_avg_distance(x_peaks))
        frame.add_feature('y_avg_max_peak_distance', self.calculate_avg_distance(y_peaks))
        frame.add_feature('z_avg_max_peak_distance', self.calculate_avg_distance(z_peaks))

    def add_mean_min_peaks(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_min'), frame.get_peaks('y_min'), frame.get_peaks('z_min')
        f_x, f_y, f_z = frame.get_function('x'), frame.get_function('y'), frame.get_function('z')
        frame.add_feature('x_min_peak_mean', self.calculate_peak_mean(x_peaks, f_x))
        frame.add_feature('y_min_peak_mean', self.calculate_peak_mean(y_peaks, f_y))
        frame.add_feature('z_min_peak_mean', self.calculate_peak_mean(z_peaks, f_z))

    def add_mean_max_peaks(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_max'), frame.get_peaks('y_max'), frame.get_peaks('z_max')
        f_x, f_y, f_z = frame.get_function('x'), frame.get_function('y'), frame.get_function('z')
        frame.add_feature('x_max_peak_mean', self.calculate_peak_mean(x_peaks, f_x))
        frame.add_feature('y_max_peak_mean', self.calculate_peak_mean(y_peaks, f_y))
        frame.add_feature('z_max_peak_mean', self.calculate_peak_mean(z_peaks, f_z))

    def add_variance_min_peaks(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_min'), frame.get_peaks('y_min'), frame.get_peaks('z_min')
        f_x, f_y, f_z = frame.get_function('x'), frame.get_function('y'), frame.get_function('z')
        frame.add_feature('x_min_peak_var', self.calculate_peak_variance(x_peaks, f_x))
        frame.add_feature('y_min_peak_var', self.calculate_peak_variance(y_peaks, f_y))
        frame.add_feature('z_min_peak_var', self.calculate_peak_variance(z_peaks, f_z))

    def add_variance_max_peaks(self, frame):
        x_peaks, y_peaks, z_peaks = frame.get_peaks('x_max'), frame.get_peaks('y_max'), frame.get_peaks('z_max')
        f_x, f_y, f_z = frame.get_function('x'), frame.get_function('y'), frame.get_function('z')
        frame.add_feature('x_max_peak_var', self.calculate_peak_variance(x_peaks, f_x))
        frame.add_feature('y_max_peak_var', self.calculate_peak_variance(y_peaks, f_y))
        frame.add_feature('z_max_peak_var', self.calculate_peak_variance(z_peaks, f_z))


    def calculate_avg_distance(self, data):
        distances = list()
        if len(data) > 0:
            i = 0
            while i < len(data) - 1:
                distance = data[i+1] - data[i]
                distances.append(distance)
                i = i+1
        return np.mean(distances)

    def calculate_peak_mean(self, data, f):
        values = f(data)
        return np.mean(values)

    def calculate_peak_variance(self, data, f):
        values = f(data)
        return np.var(values)


    # ADD DERIVATIVES

    def add_derivative_coefficients(self, frame):
        x_axis, y_axis, z_axis = frame.get_x_data(), frame.get_y_data(), frame.get_z_data()
        frame.add_derivative('x', self.calculate_derivative_coefficients(x_axis))
        frame.add_derivative('y', self.calculate_derivative_coefficients(y_axis))
        frame.add_derivative('z', self.calculate_derivative_coefficients(z_axis))

    def add_second_derivative_coefficients(self, frame):
        x_der, y_der, z_der = frame.get_derivative('x'), frame.get_derivative('y'), frame.get_derivative('z')
        frame.add_derivative('x2', self.calculate_derivative_coefficients(x_der))
        frame.add_derivative('y2', self.calculate_derivative_coefficients(y_der))
        frame.add_derivative('z2', self.calculate_derivative_coefficients(z_der))

    def calculate_derivative_coefficients(self, data):
        t = np.linspace(0, len(data)-1, len(data))
        t2 = np.linspace(1, len(data)-2, len(data))
        f = interp1d(t, data)
        return mpmath.diff(f, t2)


    # ADD PEAKS

    def add_peaks(self, frame):
        x_der, y_der, z_der = frame.get_derivative('x'), frame.get_derivative('y'), frame.get_derivative('z')
        x_peaks, y_peaks, z_peaks = self.calculate_peaks(x_der), self.calculate_peaks(y_der), self.calculate_peaks(z_der)
        f_x_der2, f_y_der2, f_z_der2 = frame.get_function('x_der2'), frame.get_function('y_der2'), frame.get_function('z_der2')

        frame.add_peaks('x_max', self.calculate_max_peaks(f_x_der2, x_peaks))
        frame.add_peaks('y_max', self.calculate_max_peaks(f_y_der2, y_peaks))
        frame.add_peaks('z_max', self.calculate_max_peaks(f_z_der2, z_peaks))
        frame.add_peaks('x_min', self.calculate_min_peaks(f_x_der2, x_peaks))
        frame.add_peaks('y_min', self.calculate_min_peaks(f_y_der2, y_peaks))
        frame.add_peaks('z_min', self.calculate_min_peaks(f_z_der2, z_peaks))

    def calculate_peaks(self, data):
        t = np.linspace(1, len(data)-2, len(data))
        f = interp1d(t, data)
        range = np.linspace(1, len(data)-2, len(data)*2)
        return self.calculate_all_roots(f, range)

    def calculate_max_peaks(self, f, peaks):
        max_peaks = list()
        for x in peaks:
            if f(x) < 0:
                max_peaks.append(x)
        return max_peaks

    def calculate_min_peaks(self, f, peaks):
        min_peaks = list()
        for x in peaks:
            if f(x) > 0:
                min_peaks.append(x)
        return min_peaks





    def calculate_apf(self, frame):
        pass

    def calculate_variance_apf(self, frame):
        pass

    def calculate_root_mean_square_of_integral(self, frame):
        pass

    def calculate_mean_min_max_sums(self, frame):
        pass
