import pytest
import numpy as np

from fairml.orthogonal_projection import audit_model
from fairml.orthogonal_projection import get_orthogonal_vector


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4


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
