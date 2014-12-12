import sys
from data_utils.framer import *
from data_utils.data_loader import *
from personal_classifier.support_vector_machine import SupportVectorMachine
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
import cPickle as pickle
import os.path

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import fft, arange


def main(argv):
    # Welcome message
    print ""
    print "Welcome to Gaiter..."
    print "First, Gaiter will load training data."

    # Check for previous data
    previous_data = False
    if os.path.isfile('data.npy'):
        # Ask user for using previous data
        previous_data_answer =  raw_input("Gaiter has discovered there is some data available from a previous session. "
                                          "This data is already framed and all features are already calculated. Do you "
                                          "want to use this data? (y/n) ")
        if previous_data_answer == 'y' or previous_data_answer == 'Y':
            print "Using the data from a previous session."
            previous_data = True
    print ""
    # Check if user wants load new data
    if not previous_data:
        # Ask user for path to directory of training data
        data_path = raw_input("Please enter the path to the directory of the training data (end your path with '/'): ")
        # Load all the data
        print "Loading data from '"+data_path+"'."
        print "..."
        print ""
        data_loader = DataLoader(data_path)
        raw_data_list = data_loader.get_raw_data()
        print str(len(raw_data_list)) + " files found."
        print "Framing all the data."
        print "..."
        print ""
        # Frame all the data
        frame_size = 128
        frame_overlap = 64
        framer = Framer(frame_size, frame_overlap)
        for raw_data in raw_data_list:
            framer.frame(raw_data)
        framed_raw_data_list = framer.get_framed_raw_data_list()
        length = 0
        for frd in framed_raw_data_list:
            length += len(frd.get_frames())
        print "Some files were not large enough for framing... "+str(len(framed_raw_data_list))+\
              " files are divided into "+str(length)+" frames."
        print ""

        print "Extracting features for all the frames. This may take a while."
        print "..."
        print ""
        # Extract features
        time_feature_extractor = TimeDomainFeatureExtractor()
        freq_feature_extractor = FrequencyDomainFeatureExtractor()
        # wav_feature_extractor = WaveletFeatureExtractor()
        bumpy_data_set = list()
        raw_data_count = 0
        for framed_raw_data in framed_raw_data_list:
            print str(round(raw_data_count/float(len(framed_raw_data_list))*100, 2))+" %"
            for frame in framed_raw_data.get_frames():
                featured_frame = time_feature_extractor.extract_features(frame)
                featured_frame = freq_feature_extractor.extract_features(featured_frame)
                # featured_frame = wav_feature_extractor.extract_features(featured_frame)
                bumpy_data_set.append(featured_frame)
            raw_data_count = raw_data_count + 1
        print "100 %"
        print ""
        print "All features are calculated. Writing all data to hard disk for later use."
        print "..."
        print ""

        flat_data_set = flatten(bumpy_data_set)
        data_set = np.array(flat_data_set)
        labels = np.array(extract_labels(bumpy_data_set))
        np.save('data', data_set)
        np.save('labels', labels)
        print "All data is written to hard drive."
        print ("The data set's dimension is " + str(data_set.shape))
    else:
        print "Loading data from previous session."
        print "..."
        data_set = np.load("data.npy")
        labels = np.load("labels.npy")
        print "All data from previous session loaded."
        print ("The data set's dimension is " + str(data_set.shape))
    print ""
    print "Gaiter will now train the walking classifier."
    print "List of walking classifiers:"
    print "1) K-means"
    print "2) K-means mini batch"
    print "3) Mean-shift"
    print ""
    walking_classifier_nb = raw_input("Please enter the number of the classifier you want to use: ")
    print "Training walking classifier."
    print "..."
    print ""
    if walking_classifier_nb == "1":
        walking_classifier = KMeans(data_set, labels)
        walking_classifier.train(4)
    if walking_classifier_nb == "2":
        walking_classifier = KMeansMiniBatch(data_set, labels)
        walking_classifier.train(4)
    if walking_classifier_nb == "3":
        walking_classifier = MeanShift(data_set, labels)
        walking_classifier.train()

    print "The walking classifier is trained."
    print ""
    print "Gaiter will now train and test the personal classifier."
    print "List of personal classifiers:"
    print "1) ADA boost"
    print "2) Random forrest"
    print "3) Support vector machine"
    print ""
    personal_classifier_nb = raw_input("Please enter the number of the classifier you want to use: ")
    print "Training personal classifier."
    print "..."
    print ""
    if personal_classifier_nb == "1":
        pass
    if personal_classifier_nb == "2":
        pass
    if personal_classifier_nb == "3":
        pass

def flatten(featured_frame_list):
    flat_list = list()
    for f_frame in featured_frame_list:
        features = f_frame.get_flat_features()
        flat_list.append(features)
    return flat_list


def extract_labels(featured_frame_list):
    classes = list()
    for featured_frame in featured_frame_list:
        l = featured_frame.get_label()
        classes.append(l)
    return classes


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

if __name__ == '__main__':
    main(sys.argv[1:])
