from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

class Validator(object):
    def __init__(self):
        pass

    def calculate_confusion_matrix(self, personal_classifier, training_set, training_labels):
        pred = personal_classifier.classify(training_set)

        # Compute confusion matrix
        cm = confusion_matrix(training_labels, pred)
        acc = accuracy_score(training_labels, pred)
        print(cm)
        print numpy.unique(training_labels)
        print(acc)
        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('img/confusion.png')