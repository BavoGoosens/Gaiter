from data_utils.frame import Frame

class Framer:

   # Initialize framer with a frame size and a frame_overlap
    def __init__(self, frame_size, frame_overlap):
        self.frame_size = frame_size
        self.frame_overlap = frame_overlap
        self.frames = list()

    def get_frame_size(self):
        return self.frame_size

    def get_frame_overlap(self):
        return self.frame_overlap

    # Return the total frame size (core + overlap)
    def get_frame_total_size(self):
        return self.get_frame_size() + self.get_frame_overlap()*2

    # Return the frames, returns an empty list if Framer.frame() hasn't been called first
    def get_frames(self):
        return self.frames

    # Split the given data into frames
    def frame(self, data):
        self.frame = list()
        data = self.delete_redundant_data_points(data)
        sample_counter = 0
        sample_index = 0
        while sample_index+self.get_frame_total_size() <= len(data):
            frame_data = list()
            while sample_counter < self.get_frame_total_size():
                frame_data.append(data[sample_index])
                sample_counter += 1
                sample_index += 1
            self.get_frames().append(Frame(frame_data, self.get_frame_size(), self.get_frame_overlap()));
            sample_counter = 0
            sample_index -= 2*self.get_frame_overlap()

    # Delete redundant points from the given data, redundant points are points that cannot be fitted into a frame
    def delete_redundant_data_points(self, data):
        redundant_points = (len(data) - self.get_frame_total_size()) % self.get_frame_size()
        k = 0
        for i in range(1,redundant_points+1):
            if k % 2 == 0: del data[0]
            else: del data[-1]
            k += 1
        return data

