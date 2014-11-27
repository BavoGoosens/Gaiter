import sys
from data_utils.framer import *
from data_utils.data_loader import *


def main(argv):
    #testDataLoader()
    #testFramer()

    frame_size = 128
    frame_overlap = 64

    data_loader = DataLoader("data/train/")
    raw_data_list = data_loader.get_raw_data()
    framer = Framer(frame_size, frame_overlap)
    framer.frame(raw_data_list)
    frames = framer.get_frames()

def testFramer():
    data = range(1,19+1)
    framer = Framer(128, 64)
    framer.frame(data)
    frames = framer.get_frames()
    print "Frames:"
    i = 1
    for frame in frames:
        print "Frame "+str(i)+": "+str(frame.get_data())
        i += 1
    print "Core data:"
    for frame in frames:
        print "Frame "+str(i)+": "+str(frame.get_core_data())
        i += 1
    print "Overlapped data:"
    for frame in frames:
        print "Frame "+str(i)+": "+str(frame.get_overlap_data())
        i += 1

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
