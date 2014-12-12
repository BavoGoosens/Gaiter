import sys
from data_utils.framer import *
from data_utils.data_loader import *
from personal_classifier.support_vector_machine import SupportVectorMachine
from feature_extraction.time_domain_feature_extractor import *
from feature_extraction.frequency_domain_feature_extractor import *
from feature_extraction.wavelet_feature_extractor import *
from personal_classifier.ada_boost import *
from personal_classifier.logistic_regression import *
from personal_classifier.random_forrests import *
from personal_classifier.support_vector_machine import *
from walking_classifier.k_means import *
from walking_classifier.k_means_mini_batch import *
from walking_classifier.mean_shift import *
from walking_classifier.db_scan import *
from monitor.timer import Timer
from scipy.sparse import csr_matrix
import monitor.time_complexity_monitor as moni
from validator import *
import matplotlib
import cPickle as pickle
import os.path
import random
import numpy as np
import math


matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import fft, arange


def main(argv):
    # Welcome message
    global walking_classifier
    print ""
    print "Welcome to Gaiter..."
    print "First, Gaiter will load training data."

    # Check for previous data
    previous_data = False
    if os.path.isfile('trainingdata.npy'):
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

        nb = int(len(framed_raw_data_list) * 0.8)
        train_raw_data_list = framed_raw_data_list[:nb]
        test_raw_data_list = framed_raw_data_list[nb:]

        # Extract features
        time_feature_extractor = TimeDomainFeatureExtractor()
        freq_feature_extractor = FrequencyDomainFeatureExtractor()
        # wav_feature_extractor = WaveletFeatureExtractor()
        print "Extracting features for training-frames. This may take a while."
        print "..."
        print ""
        bumpy_data_set = list()
        raw_data_count = 0
        for framed_raw_data in train_raw_data_list:
            print str(round(raw_data_count/float(len(train_raw_data_list))*100, 2))+" %"
            for frame in framed_raw_data.get_frames():
                featured_frame = time_feature_extractor.extract_features(frame)
                featured_frame = freq_feature_extractor.extract_features(featured_frame)
                # featured_frame = wav_feature_extractor.extract_features(featured_frame)
                bumpy_data_set.append(featured_frame)
            raw_data_count = raw_data_count + 1
        print "100 %"
        print ""
        print "Training features are calculated. Writing all data to hard disk for later use."
        print "..."
        print ""

        random.seed(0)
        random.shuffle(bumpy_data_set)
        flat_train_data_set = flatten(bumpy_data_set)
        train_data_set = np.array(flat_train_data_set)
        train_labels = np.array(extract_labels(bumpy_data_set))
        np.save('trainingdata', train_data_set)
        np.save('traininglabels', train_labels)

        print "Extracting features for testing-frames. This may take a while."
        print "..."
        print ""
        bumpy_data_set = list()
        raw_data_count = 0
        for framed_raw_data in test_raw_data_list:
            print str(round(raw_data_count/float(len(test_raw_data_list))*100, 2))+" %"
            for frame in framed_raw_data.get_frames():
                featured_frame = time_feature_extractor.extract_features(frame)
                featured_frame = freq_feature_extractor.extract_features(featured_frame)
                # featured_frame = wav_feature_extractor.extract_features(featured_frame)
                bumpy_data_set.append(featured_frame)
            raw_data_count = raw_data_count + 1
        print "100 %"
        print ""
        print "Training features are calculated. Writing all data to hard disk for later use."
        print "..."
        print ""

        random.seed(0)
        random.shuffle(bumpy_data_set)
        flat_test_data_set = flatten(bumpy_data_set)
        test_data_set = np.array(flat_test_data_set)
        test_labels = np.array(extract_labels(bumpy_data_set))
        np.save('testdata', test_data_set)
        np.save('testlabels', test_labels)

        print "All data is written to hard drive."
        print ""
        print ("The training data set's dimension is " + str(train_data_set.shape))
        print ("The testing data set's dimension is " + str(test_data_set.shape))
    else:
        print "Loading data from previous session."
        print "..."
        train_data_set = np.load("trainingdata.npy")
        train_labels = np.load("traininglabels.npy")
        test_data_set = np.load("testdata.npy")
        test_labels = np.load("testlabels.npy")
        print "All data from previous session loaded."
        print ""
        print ("The training data set's dimension is " + str(train_data_set.shape))
        print ("The testing data set's dimension is " + str(test_data_set.shape))
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
        walking_classifier = KMeans(train_data_set, train_labels)
        walking_classifier.train(4)
    if walking_classifier_nb == "2":
        walking_classifier = KMeansMiniBatch(train_data_set, train_labels)
        walking_classifier.train(4)
    if walking_classifier_nb == "3":
        walking_classifier = MeanShift(train_data_set, train_labels)
        walking_classifier.train()
    walking_data_set, walking_labels = walking_classifier.get_walking_frames()
    print "The walking classifier is trained."
    print ""
    print "Gaiter will now train and test the personal classifier."
    print "List of personal classifiers:"
    print "1) ADA boost"
    print "2) Random forrest"
    print "3) Support vector machine"
    print "4) Logistic regression"
    print ""
    personal_classifier_nb = raw_input("Please enter the number of the classifier you want to use: ")
    print "Training personal classifier."
    print "..."
    print ""
    if personal_classifier_nb == "1":
        pc = AdaBoost(walking_data_set, walking_labels)
        pc.train()
    if personal_classifier_nb == "2":
        pc = RandomForrest(walking_data_set, walking_labels)
        pc.train()
    if personal_classifier_nb == "3":
        pc = SupportVectorMachine(walking_data_set, walking_labels)
        pc.train()
    if personal_classifier_nb == "4":
        pc = LogisticRegression(walking_data_set, walking_labels)
        pc.train()
    val = Validator()
    val.calculate_confusion_matrix(pc, test_data_set, test_labels)

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
