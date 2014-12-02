class Frame:

    def __init__(self, data, size, overlap, raw_data):
        if len(data) != size + 2*overlap:
            raise AttributeError("Size of given data ("+str(len(data))+") and given size ("+str(size)+" with overlap "+str(overlap)+") are not equal.")
        self.data = data
        self.size = size
        self.overlap = overlap
        self.raw_data = raw_data

    def get_raw_data(self):
        return self.raw_data

    def get_frame_data(self):
        return self.data

    def get_t_data(self):
        t_data = list()
        for data in self.get_frame_data():
            t_data.append(data.get_time())
        return t_data

    def get_x_data(self):
        x_data = list()
        for data in self.get_frame_data():
            x_data.append(data.get_x())
        return x_data

    def get_y_data(self):
        y_data = list()
        for data in self.get_frame_data():
            y_data.append(data.get_y())
        return y_data

    def get_z_data(self):
        z_data = list()
        for data in self.get_frame_data():
            z_data.append(data.get_z())
        return z_data

    def get_size(self):
        return self.size

    def get_overlap(self):
        return self.overlap

    def get_core_data(self):
        return self.get_frame_data()[self.get_overlap():-self.get_overlap()]

    def get_overlap_data(self):
        overlap1 = self.get_frame_data()[0:self.get_overlap()]
        overlap2 = self.get_frame_data()[self.get_overlap()+self.get_size():len(self.get_frame_data())]
        return [overlap1, overlap2]