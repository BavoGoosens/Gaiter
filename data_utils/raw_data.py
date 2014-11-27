

class RawData:
    def __init__(self, label, t, x, y, z, file_path):
        self.label = label
        self.time = t
        self.x = x
        self.y = y
        self.z = z
        self.path = file_path

    def get_label(self):
        return self.label

    def get_time(self):
        return self.time

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_path(self):
        return self.path






