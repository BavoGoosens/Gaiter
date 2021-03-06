import time


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.s = self.end - self.start
        self.ms = self.s * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.ms