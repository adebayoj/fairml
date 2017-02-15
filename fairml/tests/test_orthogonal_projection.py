from __future__ import division


import pytest
import numpy as np
from random import randint

from fairml.orthogonal_projection import audit_model
from fairml.orthogonal_projection import get_orthogonal_vector

from fairml.utils import mse
from fairml.utils import accuracy
from fairml.utils import detect_feature_sign

from fairml.perturbation_strategies import constant_zero


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