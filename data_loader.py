import os
import csv
from raw_data import RawData


class DataLoader:
    """
    Creates a new DataLoader which scans and loads all csv files in the provided directory
    """
    def __init__(self, file_directory):
        self.data = list()
        self.scan(file_directory)

    """
    This method scans for csv files in the directory structure and loads them into the system
    """
    def scan(self, file_directory):
        for file in os.listdir(file_directory):
            if file.endswith(".csv"):
                name = file.split(".")[0]
                path = file_directory + file
                self.load(name, path)

    """
    This method creates the actual raw data object and stores it internally
    """
    def load(self, name, location):
        label = name.split("_")[-1]
        t = list()
        x = list()
        y = list()
        z = list()
        reader = csv.reader(open(location, 'rb'))
        for row in reader:
            t.append(row[0])
            x.append(row[1])
            y.append(row[2])
            z.append(row[3])
        raw_data = RawData(label, t, x, y , z, location)
        self.data.append(raw_data)

    def get_raw_data(self):
        return self.data