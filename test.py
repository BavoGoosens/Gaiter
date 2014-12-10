from data_utils.data_loader import *
from data_utils.framer import *
from feature_extraction.time_domain_feature_extractor import *

import numpy as np

data_loader = DataLoader("data/train/")
raw_data_list = data_loader.get_raw_data()
framer = Framer(128, 64)
for raw_data in raw_data_list:
    framer.frame(raw_data)

framed_raw_data_list = framer.get_framed_raw_data_list()
frame = framed_raw_data_list[0].get_frames()[0]

feature_extractor = TimeDomainFeatureExtractor()
frame = feature_extractor.extract_features(frame)

t = np.linspace(0,255,256)
