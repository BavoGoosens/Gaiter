from collections import defaultdict

measurements = defaultdict(list)

"""
post a measurement
time = spent time in ms
categories = {loading, framing, feature extraction, walking classifier, personal classifier}
"""


def post(time, category, msg):
    measurements(category).append(msg, time)


def get_measurements(category):
    return measurements(category)