from data_utils.data_loader import *
from data_utils.framer import *
from feature_extraction.time_domain_feature_extractor import *
from feature_extraction.frequency_domain_feature_extractor import *
from feature_extraction.wavelet_feature_extractor import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_loader = DataLoader("data/train/")
raw_data_list = data_loader.get_raw_data()
framer = Framer(128, 64)
for raw_data in raw_data_list:
    framer.frame(raw_data)

framed_raw_data_list = framer.get_framed_raw_data_list()
frame = framed_raw_data_list[3].get_frames()[7]

feature_extractor = TimeDomainFeatureExtractor()
frame = feature_extractor.extract_features(frame)
feature_extractor = FrequencyDomainFeatureExtractor()
frame = feature_extractor.extract_features(frame)
feature_extractor = WaveletFeatureExtractor()
frame = feature_extractor.extract_features(frame)

features = frame.get_all_features()
for key in sorted(features.keys()):
    print str(key)+": "+str(features[key])
print ""
print ""
print "Total features: "+str(len(features))
t = np.linspace(0,255,256)
t2 = np.linspace(1,254,256)
plt.plot(t, frame.get_function('x')(t))
plt.plot(t2, frame.get_function('x_der')(t2))
peaks = frame.get_peaks('x_max')
plt.plot(peaks, frame.get_function('x')(peaks), marker='o', linestyle='')
peaks = frame.get_peaks('x_min')
plt.plot(peaks, frame.get_function('x')(peaks), marker='o', linestyle='')
plt.savefig('plotderiv.png');