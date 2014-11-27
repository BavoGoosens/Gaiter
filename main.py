import sys

import matplotlib.pyplot as plt

from data_utils.data_loader import *


def main(argv):
    data_loader = DataLoader("data/train/")
    data = data_loader.get_raw_data()
    entry = data[1]
    t = entry.get_time()
    x = entry.get_x_axis()
    y = entry.get_y_axis()
    z = entry.get_z_axis()
    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, z)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])