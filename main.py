import sys
from data_utils.framer import *
from data_utils.data_loader import *
from feature_extraction.time_domain_feature_extractor import *
from feature_extraction.frequency_domain_feature_extractor import *
from feature_extraction.wavelet_feature_extractor import *
from walking_classifier.k_means import *
from walking_classifier.k_means_mini_batch import *
from walking_classifier.mean_shift import *
from walking_classifier.db_scan import *
from monitor.timer import Timer
from scipy.sparse import csr_matrix
import monitor.time_complexity_monitor as moni
import matplotlib
import pickle

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import fft, arange


def main(argv):
    # test_framer()

    frame_size = 128
    frame_overlap = 64

    data_present = False

    if not data_present:
        # Load all the data
        with Timer() as t:
            data_loader = DataLoader("data/train/")
            raw_data_list = data_loader.get_raw_data()
        moni.post(t.ms, "loading", "all training data scanned and loaded")
        print("nb of files found " + str(len(raw_data_list)))

        # Frame all the data
        with Timer() as t:
            framer = Framer(frame_size, frame_overlap)
            for raw_data in raw_data_list:
                framer.frame(raw_data)
            framed_raw_data_list = framer.get_framed_raw_data_list()
        moni.post(t.ms, "framing", "Framed all data")
        length = 0
        for frd in framed_raw_data_list:
            length += len(frd.get_frames())
        print("nb of usable frames " + str(length))

        # create the concrete feature extractors
        time_feature_extractor = TimeDomainFeatureExtractor()
        freq_feature_extractor = FrequencyDomainFeatureExtractor()
        wav_feature_extractor = WaveletFeatureExtractor()
        # Extract features from the frames
        bumpy_data_set = list()
        raw_data_count = 1
        for framed_raw_data in framed_raw_data_list:
            frame_count = 0
            length = len(framed_raw_data.get_frames())
            for frame in framed_raw_data.get_frames():
                print str(raw_data_count)+": "+str(round((frame_count/float(length))*100, 2))+" %"
                featured_frame = time_feature_extractor.extract_features(frame)
                featured_frame = freq_feature_extractor.extract_features(featured_frame)
                # featured_frame = wav_feature_extractor.extract_features(featured_frame)
                bumpy_data_set.append(featured_frame)
                frame_count = frame_count + 1
            print str(raw_data_count)+": 100 %"
            print ""
            raw_data_count = raw_data_count + 1
        with open('bin.dat', 'wb') as f:
            pickle.dump(bumpy_data_set, f)
    else:
        with open('bin.dat') as f:
            bumpy_data_set = pickle.load(f)
        print(bumpy_data_set)

        km = KMeans(bumpy_data_set)
        kmmb = KMeansMiniBatch(bumpy_data_set)
        ms = MeanShift(bumpy_data_set)
        db = DBScan(bumpy_data_set)

        db.train()
        ms.train()
        kmmb.train(nb_clusters=4)
        km.train(nb_clusters=4)

        ms.get_walking_frames()


def test_framer():
    data = range(1, 100 + 1)
    framer = Framer(6, 2)
    framer.frame(data)
    frames = framer.get_frames()
    print "Frames:"
    i = 1
    for frame in frames:
        print "Frame " + str(i)
        print "Data: " + str(frame.get_data())
        print "Core data: " + str(frame.get_core_data())
        print "Overlapped data: " + str(frame.get_overlap_data())
        i += 1


if __name__ == '__main__':
    main(sys.argv[1:])
