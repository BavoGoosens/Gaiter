import sys
from data_utils.framer import *
from data_utils.data_loader import *
from feature_extraction.time_domain_feature_extractor import *
from feature_extraction.frequency_domain_feature_extractor import *
from feature_extraction.wavelet_feature_extractor import *
from monitor.timer import Timer
import monitor.time_complexity_monitor as moni
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import fft, arange


def main(argv):
    # testFramer()

    frame_size = 128
    frame_overlap = 64

    #Load all the data
    with Timer() as t:
        data_loader = DataLoader("data/train/")
        raw_data_list = data_loader.get_raw_data()
    moni.post(t.ms, "loading", "all training data scanned and loaded")
    print("nb of files found " + str(len(raw_data_list)))

    #Frame all the data
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
    wav_feature_extractor = WaveletFeatureExtractor();
    # Extract features from the frames
    dataset = list()
    for framed_raw_data in framed_raw_data_list:
        for frame in framed_raw_data.get_frames():
            featured_frame = time_feature_extractor.extract_features(frame)
            featured_frame = freq_feature_extractor.extract_features(featured_frame)
            # featured_frame = wav_feature_extractor.extract_features(featured_frame)
            dataset.append(featured_frame.get_all_features())
    print("Dataset length " + str(len(dataset)))

    framed_raw_data = framed_raw_data_list[10]
    frames = framed_raw_data.get_frames()
    frame = frames[8]
    feature_extractor = TimeDomainFeatureExtractor()
    featured_frame = feature_extractor.extract_features(frame)
    feature_extractor = FrequencyDomainFeatureExtractor()
    featured_frame = feature_extractor.extract_features(featured_frame)
    features = featured_frame.get_all_features()

    for key in sorted(features.keys()):
        if isinstance(features[key], np.ndarray):
            print str(key) + " length: " + str(len(features[key]))
        else:
            print str(key) + ": " + str(features[key])
    plt.plot(featured_frame.get_t_data(), featured_frame.get_y_data())
    plt.savefig('test.png')
    plt.clf()
    y = featured_frame.get_x_data()
    for a in y:
        print a
    print featured_frame.get_feature('x_spectral_cos')


def testFramer():
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
