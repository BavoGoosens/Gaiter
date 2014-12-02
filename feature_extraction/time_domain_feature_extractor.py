from feature_extractor import FeatureExtractor

# TODO
class TimeDomainFeatureExtractor(FeatureExtractor):
    def __init__(self, frame):
        self.frame = frame

    def set_frame(self, frame):
        self.frame = frame

    def extract_features(self):
        self.calculate_mean()
        self.calculate_integral()
        self.calculate_kurtosis()
        self.calculate_abs_mean()
        self.calculate_variance()
        self.calculate_minimum()
        self.calculate_maximum()
        self.calculate_abs_minimum()
        self.calculate_abs_maximum()
        self.calculate_avg_peak_distance()
        self.calculate_peak_mean()
        self.calculate_variance_in_peak()
        self.calculate_apf()
        self.calculate_variance_apf()
        self.calculate_root_mean_square()
        self.calculate_min_max_difference()
        self.calculate_percentiles()
        self.calculate_skewness()
        self.calculate_median()
        self.calculate_root_mean_square_of_integral()
        self.calculate_mean_min_max_sums()

    def calculate_mean(self):
        pass

    def calculate_integral(self):
        pass

    def calculate_kurtosis(self):
        pass

    def calculate_abs_mean(self):
        pass

    def calculate_variance(self):
        pass

    def calculate_minimum(self):
        pass

    def calculate_maximum(self):
        pass

    def calculate_abs_minimum(self):
        pass

    def calculate_abs_maximum(self):
        pass

    def calculate_avg_peak_distance(self):
        pass

    def calculate_peak_mean(self):
        pass

    def calculate_variance_in_peak(self):
        pass

    def calculate_apf(self):
        pass

    def calculate_variance_apf(self):
        pass

    def calculate_root_mean_square(self):
        pass

    def calculate_min_max_difference(self):
        pass

    def calculate_percentiles(self):
        pass

    def calculate_skewness(self):
        pass

    def calculate_median(self):
        pass

    def calculate_root_mean_square_of_integral(self):
        pass

    def calculate_mean_min_max_sums(self):
        pass