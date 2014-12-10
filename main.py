import sys
from data_utils.framer import *
from data_utils.data_loader import *
from feature_extraction.time_domain_feature_extractor import *
from feature_extraction.frequency_domain_feature_extractor import *
from feature_extraction.wavelet_feature_extractor import *
from monitor.timer import Timer
from scipy.sparse import csr_matrix
import monitor.time_complexity_monitor as moni
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import fft, arange


def main(argv):
    # test_framer()

    frame_size = 128
    frame_overlap = 64

    data_present = True

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
        for framed_raw_data in framed_raw_data_list:
            for frame in framed_raw_data.get_frames():
                featured_frame = time_feature_extractor.extract_features(frame)
                featured_frame = freq_feature_extractor.extract_features(featured_frame)
                # featured_frame = wav_feature_extractor.extract_features(featured_frame)
                bumpy_data_set.append(featured_frame)
        flat_data_set = flatten(bumpy_data_set)
        data_set = csr_matrix(flat_data_set)
        save_sparse_csr("data", data_set)
        print ("The data set dimension is " + str(data_set.shape))
    else:
        data_set = load_sparse_csr("data.npz")
        print ("The data set dimension is " + str(data_set.shape))


def flatten(featured_frame_list):
    flat_list = list()
    for f_frame in featured_frame_list:
        features = f_frame.get_flat_features()
        flat_list.append(features)
    return flat_list


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


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


if __name__ == '__main__':
    main(sys.argv[1:])
