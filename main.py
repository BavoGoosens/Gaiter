import sys
from data_utils.framer import *
from data_utils.data_loader import *


def main(argv):
    #testFramer()

    frame_size = 128
    frame_overlap = 64

    data_loader = DataLoader("data/train/")
    raw_data_list = data_loader.get_raw_data()
    framer = Framer(frame_size, frame_overlap)
    for raw_data in raw_data_list:
        framer.frame(raw_data)
    framed_raw_data_list = framer.get_framed_raw_data_list()
    framed_raw_data = framed_raw_data_list[0]
    frames = framed_raw_data.get_frames()
    frame = frames[0]
    print frame.get_x_data()
    print len(frame.get_x_data())


def testFramer():
    data = range(1,100+1)
    framer = Framer(6, 2)
    framer.frame(data)
    frames = framer.get_frames()
    print "Frames:"
    i = 1
    for frame in frames:
        print "Frame "+str(i)
        print "Data: "+str(frame.get_data())
        print "Core data: "+str(frame.get_core_data())
        print "Overlapped data: "+str(frame.get_overlap_data())
        i += 1

if __name__ == '__main__':
    main(sys.argv[1:])
