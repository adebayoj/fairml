from __future__ import division


import pytest
import numpy as np
from random import randint

from fairml.orthogonal_projection import audit_model
from fairml.orthogonal_projection import get_orthogonal_vector

from fairml.utils import mse
from fairml.utils import accuracy
from fairml.utils import gini
from fairml.utils import detect_feature_sign

from fairml.perturbation_strategies import constant_zero


# let's define a black-box function
def black_box_function(input_data):
    if not (input_data.shape[1] == weights.shape[0]):
        raise Exception("problem, misaligned dimensions")
    output = np.dot(input_data, weights)
    return output


def test_orthogonal_projection(number_of_tries=20, size=10000):
    """Orthogonal projection function. """
    for i in range(number_of_tries):

        a = np.random.normal(0, 1, size)
        b = np.random.normal(0, 1, size)
        c = np.random.binomial(10, 0.1, size)
        d = np.random.uniform(0, 10, size)

        # normal-normal check
        orth_b = get_orthogonal_vector(a, b)
        assert np.dot(orth_b, a) < 1e-8

        # normal- normal check
        ortho_c = get_orthogonal_vector(a, c)
        assert np.dot(ortho_c, a) < 1e-8

        # normal - uniform check
        ortho_d = get_orthogonal_vector(a, d)
        assert np.dot(ortho_d, a) < 1e-8


def test_mse():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    test_mse = mse(y_true, y_pred)
    assert test_mse == 0.375


def test_accuracy():
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]

    test_acc = accuracy(y_pred, y_true)
    print(test_acc)
    assert test_acc == 0.5

def test_gini():
    y_pred = [0, 0, 0, 1]
    y_true = [0, 0, 1, 1]

    test_gini= gini(y_true, y_pred)
    print(test_gini)
    assert test_gini == 0.5
