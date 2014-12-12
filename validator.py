from sklearn.metrics import confusion_matrix


class Validator(object):
    def __init__(self):
        pass

    def calculate_confusion_matrix(self, personal_classifier, training_set):
        personal_classifier.get_la
        pred = personal_classifier.get_classifier.classify(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(cm)
