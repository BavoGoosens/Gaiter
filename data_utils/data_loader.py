import os
import csv

from data_utils.raw_data import RawData


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
    def load(self, name, path):
        label = name.split("_")[-1]
        reader = csv.reader(open(path, 'rb'))
        for row in reader:
            t = row[0]
            x = row[1]
            y = row[2]
            z = row[3]
            raw_data = RawData(label, t, x, y , z, path)
            self.data.append(raw_data)

    def get_raw_data(self):
        return self.data