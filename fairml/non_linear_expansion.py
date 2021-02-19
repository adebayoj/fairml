from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from collections import defaultdict
from random import randint
import six


def return_non_linear_transformation(v1, poly, log, square_root,
                                     exponential, sin, cos):
    A = np.zeros((v1.shape[0], 1))
    A[:, 0] = v1
    if poly > 1:
        for i in range(2, poly + 1):
            current_power = v1**i
            current_power = np.reshape(current_power,
                                       (current_power.shape[0], 1))
            A = np.append(A, current_power, axis=1)

    if square_root:
        sqrt = np.sqrt(v1 + np.abs(min(v1)) + 1)
        sqrt = np.reshape(sqrt, (sqrt.shape[0], 1))
        A = np.append(A, sqrt, axis=1)

    if exponential:
        exp_term = np.exp(v1 - np.max(v1))
        exp_term = np.reshape(exp_term, (exp_term.shape[0], 1))
        A = np.append(A, exp_term, axis=1)

    if log:
        # shift the entire vector by the minimum value, which is +1
        log_term = np.log(v1 + np.abs(min(v1) + 1))
        log_term = np.reshape(log_term, (log_term.shape[0], 1))
        A = np.append(A, log_term, axis=1)

    if sin:
        sin_term = np.sin(v1)
        sin_Term = np.reshape(sin_term, (sin_term.shape[0], 1))
        A = np.append(A, sin_term, axis=1)

    if cos:
        cos_term = np.cos(v1)
        cos_term = np.reshape(cos_term, (cos_term.shape[0], 1))
        A = np.append(A, cos_term, axis=1)

    return A
