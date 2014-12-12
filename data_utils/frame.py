class Frame(object):

    # Initialize the frame
    # The given size and overlap must be compatible with the size of the data list
    def __init__(self, data, size, overlap, raw_data):
        if len(data) != size + 2*overlap:
            raise AttributeError("Size of given data ("+str(len(data))+") and given size ("+str(size)+" with overlap "+str(overlap)+") are not equal.")
        self.data = data
        self.size = size
        self.overlap = overlap
        self.raw_data = raw_data

    # Return a raw data object that belongs to the frame
    def get_raw_data(self):
        return self.raw_data

    def get_label(self):
        return self.raw_data.get_label()

    # Return all the data in the frame as a list of data row objects
    def get_frame_data(self):
        return self.data

    # Return the time-axis data in the frame as a list of doubles
    def get_t_data(self):
        t_data = list()
        for data in self.get_frame_data():
            t_data.append(data.get_time())
        return t_data

    # Return the x-axis data in the frame as a list of doubles
    def get_x_data(self):
        x_data = list()
        for data in self.get_frame_data():
            x_data.append(data.get_x())
        return x_data

    # Return the y-axis data in the frame as a list of doubles
    def get_y_data(self):
        y_data = list()
        for data in self.get_frame_data():
            y_data.append(data.get_y())
        return y_data

    # Return the z-axis data in the frame as a list of doubles
    def get_z_data(self):
        z_data = list()
        for data in self.get_frame_data():
            z_data.append(data.get_z())
        return z_data

    # Return the size of the frame
    def get_size(self):
        return self.size

    # Return the overlap size of the frame
    def get_overlap(self):
        return self.overlap

    # Return the core data in the frame as a list of data row objects
    def get_core_data(self):
        return self.get_frame_data()[self.get_overlap():-self.get_overlap()]

    # Return the overlapped data in the frame as a list of data row objects
    def get_overlap_data(self):
        overlap1 = self.get_frame_data()[0:self.get_overlap()]
        overlap2 = self.get_frame_data()[self.get_overlap()+self.get_size():len(self.get_frame_data())]
        return [overlap1, overlap2]