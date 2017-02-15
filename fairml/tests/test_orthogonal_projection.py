import pytest
import numpy as np 

from fairml.orthogonal_projection import audit_model
from fairml.orthogonal_projection import get_orthogonal_vector

# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4

def test_constant_zero_perturbation_strategy(number_of_tries=20):

    #do a few tries
    for i in range(number_of_tries):

        a = np.random.normal(0, 1, 10000)
        b = np.random.binomial(10, 0.1, 10000)

        orth_b = get_orthogonal_vector(a, b)

        assert np.dot(orth_b, a) < 1e-6




