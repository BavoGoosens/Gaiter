from data_utils.frame import Frame

class Framer:
    def __init__(self, frame_size, frame_overlap):
        self.frame_size = frame_size
        self.frame_overlap = frame_overlap
        self.frames = list()

    def get_frame_size(self):
        return self.frame_size

    def get_frame_overlap(self):
        return self.frame_overlap

    def get_frame_total_size(self):
        return self.get_frame_size() + self.get_frame_overlap()*2

    def get_frames(self):
        return self.frames

    def frame(self, data):

        sample_counter = 0
        sample_index = 0
        frame = Frame()

        while sample_index+self.get_frame_total_size() < len(data):

            while sample_counter < self.get_frame_total_size():
                if sample_index >= 0:
                    frame.add_data(data[sample_index])
                sample_counter += 1
                sample_index += 1

            self.get_frames().append(frame)
            frame = Frame()
            sample_counter = 0
            sample_index -= 2*self.get_frame_overlap()

