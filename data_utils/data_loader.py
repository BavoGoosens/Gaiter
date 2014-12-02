import os
import csv
from collections import defaultdict

from data_utils.raw_data import RawData
from data_utils.data_row import DataRow


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
        data_rows = list()
        for row in reader:
            t = row[0]
            x = row[1]
            y = row[2]
            z = row[3]
            data_row = DataRow(t, x, y, z)
            data_rows.append(data_row)
        raw_data = RawData(label, path, data_rows)
        self.data.append(raw_data)

    def get_raw_data(self):
        return self.data