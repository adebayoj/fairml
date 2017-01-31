import numpy as np


name_dictionary = {}


def constant_zero(X, column_number, random_sample):
    return 0.0


def constant_median(X, column_number, random_sample):
    return np.median(X[:, column_number])


def random_sample(X, column_number, random_sample):
    return random_sample[random_sample]

