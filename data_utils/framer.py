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
        redundant_points = self.calculate_redundant_data_points(data)
        k = 0
        for i in range(1,redundant_points+1):
            if k % 2 == 0: del data[0]
            else: del data[-1]
            k += 1
        sample_counter = 0
        sample_index = 0
        while sample_index+self.get_frame_total_size() < len(data)+1:
            frame_data = list()
            while sample_counter < self.get_frame_total_size():
                frame_data.append(data[sample_index])
                sample_counter += 1
                sample_index += 1
            self.get_frames().append(Frame(frame_data, self.get_frame_size(), self.get_frame_overlap()));
            sample_counter = 0
            sample_index -= 2*self.get_frame_overlap()

    def calculate_redundant_data_points(self, data):
        return (len(data) - self.get_frame_total_size()) % self.get_frame_size()

