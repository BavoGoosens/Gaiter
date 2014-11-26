import abc


class FeatureExtractor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract_features(self):
        """ This method needs to be implemented in all the subclasses """

