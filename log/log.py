
class Log(object):

    def __init__(self):
        self.data_loading = dict()
        self.framing = dict()
        self.time_domain_feature_extraction = dict()
        self.frequency_domain_feature_extraction = dict()
        self.walking_classifying = dict()
        self.personal_classifying = dict()

    def add_log(self, description, time):
        self.logs[description] = time