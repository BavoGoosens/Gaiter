

class RawData:
    def __init__(self, label, t, x, y, z, file_path):
        self.label = label
        self.time = t
        self.x = x
        self.y = y
        self.z = z
        self.path = file_path

    def get_time(self):
        return self.time[1:]

    def get_x_axis(self):
        return self.x[1:]

    def get_y_axis(self):
        return self.y[1:]

    def get_z_axis(self):
        return self.z[1:]






