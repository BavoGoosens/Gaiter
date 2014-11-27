import sys
from data_loader import *
from framer import Framer
"""import matplotlib.pyplot as plt"""

from data_utils.data_loader import *


def main(argv):
    testDataLoader()
    #testFramer()
"""    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, z)
    plt.show()"""

def testFramer():
    data = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "Test7", "Test8", "Test9", "Test10", "Test11", "Test12", "Test13", "Test14", "Test15", "Test16", "Test17", "Test18", "Test19", "Test20"]
    framer = Framer(3, 2)
    framer.frame(data)
    frames = framer.get_frames()
    for frame in frames:
        print frame.get_data()

def testFramer2(data):
    framer = Framer(128, 64)
    framer.frame(data)
    frames = framer.get_frames()
    i = 1
    for frame in frames:
        print "Frame number "+str(i)
        i += 1
        for data in frame.get_data():
            print data.get_time()

def testDataLoader():
    data_loader = DataLoader("data/train/")
    raw_data = data_loader.get_raw_data()
    """for element in raw_data:
        print "label: "+str(element.get_label())
        print "time: "+str(element.get_time())
        print "x: "+str(element.get_x())
        print "y: "+str(element.get_y())
        print "z: "+str(element.get_z())"""
    testFramer2(raw_data)



if __name__ == '__main__':
    main(sys.argv[1:])
