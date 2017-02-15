from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# import dictionary with perturbation strategies.
from .perturbation_strategies import perturbation_strategy_dictionary


def mse(y, y_hat):
    """ function to calculate mse between to numpy vectors """

    y = np.array(y)
    y_hat = np.array(y_hat)

    y_hat = np.reshape(y_hat, (y_hat.shape[0],))
    y = np.reshape(y, (y.shape[0],))

    diff = y - y_hat
    diff_squared = np.square(diff)
    mse = np.mean(diff_squared)

    return mse


def accuracy(y, y_hat):
    """ function to calculate accuracy of y_hat given y"""
    y = np.array(y)
    y_hat = np.array(y_hat)

    y = y.astype(int)
    y_hat = y_hat.astype(int)

    y_hat = np.reshape(y_hat, (y_hat.shape[0],))
    y = np.reshape(y, (y.shape[0],))

    equal = (y == y_hat)
    accuracy = np.sum(equal) / y.shape[0]

    return accuracy


def replace_column_of_matrix(X, col_num, random_sample,
                             ptb_strategy):
    """
    Arguments: data matrix, n X k
    random sample: row of data matrix, 1 X k
    column number: 0 <-> k-1

    replace all elements of X[column number] X
    with random_sample[column_number]
    """

    # need to implement random permutation.
    # need to implement perturbation strategy as a function
    # need a distance metrics file.
    # this probably does not work right now, I need to go through to fix.
    if col_num >= random_sample.shape[0]:
        raise ValueError("column {} entered. Column # should be"
                         "less than {}".format(col_num,
                                               random_sample.shape[0]))

    # select the specific perturbation function chosen
    # obtain value from that function
    val_chosen = perturbation_strategy_dictionary[ptb_strategy](X,
                                                                col_num,
                                                                random_sample)
    constant_array = np.repeat(val_chosen, X.shape[0])
    X[:, col_num] = constant_array

    return X


def detect_feature_sign(predict_function, X, col_num):

    normal_output = predict_function(X)
    column_range = X[:, col_num].max() - X[:, col_num].min()

    X[:, col_num] = X[:, col_num] + np.repeat(column_range, X.shape[0])
    new_output = predict_function(X)

    diff = new_output - normal_output
    total_diff = np.mean(diff)

    if total_diff >= 0:
        return 1
    else:
        return -1


def main():
    pass


if __name__ == '__main__':
    main()
