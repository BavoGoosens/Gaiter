class Frame:

    def __init__(self, data, size, overlap):
        if len(data) != size + 2*overlap:
            raise AttributeError("Size of given data ("+str(len(data))+") and given size ("+str(size)+" with overlap "+str(overlap)+") are not equal.")
        self.data = data
        self.size = size
        self.overlap = overlap


    def get_data(self):
        return self.data

    def get_size(self):
        return self.size

    def get_overlap(self):
        return self.overlap

    def get_core_data(self):
        return self.get_data()[self.get_overlap():-self.get_overlap()]

    def get_overlap_data(self):
        overlap1 = self.get_data()[0:self.get_overlap()]
        overlap2 = self.get_data()[self.get_overlap()+self.get_size():len(self.get_data())]
        return [overlap1, overlap2]