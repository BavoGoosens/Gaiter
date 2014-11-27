import sys
from data_utils.framer import *
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
    print "Frames:"
    for frame in frames:
        print frame.get_data()
    print "Core data:"
    for frame in frames:
        print frame.get_core_data()
    print "Overlapped data:"
    for frame in frames:
        print frame.get_overlap_data()

def testFramer2(data):
    framer = Framer(128, 64)
    framer.frame(data)
    frames = framer.get_frames()
    frame = frames[1]
    print "Frame 1:"
    print "Data:"
    for data in frame.get_data():
        print data.get_x()
    print "Core data:"
    for data in frame.get_core_data():
        print data.get_x()
    print "Overlapped data:"
    for data in frame.get_overlap_data():
        for overlap in data:
            print overlap.get_x()
    frame = frames[2]
    print "Frame 2:"
    print "Data:"
    for data in frame.get_data():
        print data.get_x()
    print "Core data:"
    for data in frame.get_core_data():
        print data.get_x()
    print "Overlapped data:"
    for data in frame.get_overlap_data():
        for overlap in data:
            print overlap.get_x()
        print "----------------"

def testDataLoader():
    data_loader = DataLoader("data/train/")
    raw_data = data_loader.get_raw_data()
    testFramer2(raw_data)



if __name__ == '__main__':
    main(sys.argv[1:])
