

class PersonalClassifier(object):
    def __init__(self, featured_frames):
        self.frames = featured_frames
        self.data_set = self.flatten(self.frames)
        self.labels = self.extract_labels(self.frames)

    def flatten(self, featured_frame_list):
        flat_list = list()
        for f_frame in featured_frame_list:
            features = f_frame.get_flat_features()
            flat_list.append(features)
        return flat_list

    def extract_labels(self, featured_frame_list):
        classes = list()
        for featured_frame in featured_frame_list:
            l = featured_frame.get_label()
            classes.append(l)
        return classes
