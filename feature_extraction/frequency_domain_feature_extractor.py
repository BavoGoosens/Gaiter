from feature_extractor import *
from data_utils.featured_frame import *

import scipy.fftpack as ff
import numpy as np

class FrequencyDomainFeatureExtractor(FeatureExtractor):

    def extract_features(self, frame):
        if not isinstance(frame, FeaturedFrame):
            frame = FeaturedFrame(frame)
        self.add_spectral_coefficients(frame)
        self.add_mel_scale_coefficients(frame)
        self.add_spectral_mean(frame)
        self.add_spectral_energy(frame)
        self.add_dc_component(frame)
        return frame

    def calculate_spectral_coefficients(self, data):
        return ff.fft(data)

    def calculate_mel_scale_coefficients(self, frame, data):
        complex_spectrum = frame.get_coefficients('x_spectral_cos')
        power_spectrum = abs(complex_spectrum) ** 2
        filtered_spectrum = np.dot(power_spectrum, self.melFilterBank(256))
        log_spectrum = np.log(filtered_spectrum)
        dctSpectrum = ff.dct(log_spectrum, type=2)
        print (dctSpectrum)

    def add_spectral_coefficients(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        frame.add_coefficients('x_spectral_cos', self.calculate_spectral_coefficients(x_axis))
        frame.add_coefficients('y_spectral_cos', self.calculate_spectral_coefficients(y_axis))
        frame.add_coefficients('z_spectral_cos', self.calculate_spectral_coefficients(z_axis))

    def add_mel_scale_coefficients(self, frame):
        x_axis = frame.get_x_data()
        y_axis = frame.get_y_data()
        z_axis = frame.get_z_data()
        if self.has_coefficients(frame, 'spectral'):
            frame.add_coefficients('x_MFCC', self.calculate_mel_scale_coefficients(frame, x_axis))
            frame.add_coefficients('y_MFCC', self.calculate_mel_scale_coefficients(frame, y_axis))
            frame.add_coefficients('z_MFCC', self.calculate_mel_scale_coefficients(frame, z_axis))
        else:
            self.add_spectral_coefficients(frame)

    def add_linear_coefficients(self):
        pass

    def add_spectral_mean(self, frame):
        if self.has_coefficients(frame, 'spectral'):
            frame.add_feature('x_spectral_mean', np.mean(frame.get_coefficients('x_spectral_cos')))
            frame.add_feature('y_spectral_mean', np.mean(frame.get_coefficients('y_spectral_cos')))
            frame.add_feature('z_spectral_mean', np.mean(frame.get_coefficients('z_spectral_cos')))
        else:
            self.add_spectral_coefficients(frame)

    def add_spectral_energy(self, frame):
        if self.has_coefficients(frame, 'spectral'):
            frame.add_feature('x_spectral_energy', np.mean(np.power(frame.get_coefficients('x_spectral_cos'), 2)))
            frame.add_feature('y_spectral_energy', np.mean(np.power(frame.get_coefficients('y_spectral_cos'), 2)))
            frame.add_feature('z_spectral_energy', np.mean(np.power(frame.get_coefficients('z_spectral_cos'), 2)))
        else:
            self.add_spectral_coefficients(frame)

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

    # checks if the type of coefficients are already present
    def has_coefficients(self, frame, type):
        if type == 'spectral':
            return len(frame.get_coefficients('x_spectral_cos')) > 0
        elif type == 'mel':
            return len(frame.get_coefficients('x_MFCC')) > 0
        else:
            return False

    def melFilterBank(self, blockSize):
        numCoefficients = 13 # choose the size of mfcc array
        minHz = 0
        maxHz = 50
        numBands = int(numCoefficients)
        maxMel = int(self.freqToMel(maxHz))
        minMel = int(self.freqToMel(minHz))

        # Create a matrix for triangular filters, one row per filter
        filterMatrix = np.zeros((numBands, blockSize))

        melRange = np.array(xrange(numBands + 2))

        melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

        # each array index represent the center of each triangular filter
        aux = np.log(1 + 1000.0 / 700.0) / 1000.0
        aux = (np.exp(melCenterFilters * aux) - 1) / 22050
        aux = 0.5 + 700 * blockSize * aux
        aux = np.floor(aux)  # round down
        centerIndex = np.array(aux, int)  # Get int values

        for i in xrange(numBands):
            start, centre, end = centerIndex[i:i + 3]
            k1 = np.float32(centre - start)
            k2 = np.float32(end - centre)
            up = (np.array(xrange(start, centre)) - start) / k1
            down = (end - np.array(xrange(centre, end))) / k2

            filterMatrix[i][start:centre] = up
            filterMatrix[i][centre:end] = down
        return filterMatrix.transpose()

    def freqToMel(self, freq):
        return 1127.01048 * np.math.log(1 + freq / 700.0)

    def melToFreq(self, mel):
        return 700 * (np.math.exp(mel / 1127.01048 - 1))
