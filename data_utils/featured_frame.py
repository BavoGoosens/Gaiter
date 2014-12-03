from frame import *

class FeaturedFrame(Frame):

    def __init__(self, frame):
        super(FeaturedFrame, self).__init__(frame.get_frame_data(), frame.get_size(), frame.get_overlap(), frame.get_raw_data())
        self.features = dict()

    def add_feature(self, name, value):
        self.features[name] = value

    def get_feature(self, name):
        return self.features[name]

    def get_all_features(self):
        return self.features