import numpy as np


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
    accuracy = np.sum(equal)/y.shape[0]

    return accuracy


def replace_column_of_matrix(X, column_number, random_sample,
                             perturbation_strategy):
    """
    Arguments: data matrix, n X k
    random sample: row of data matrix, 1 X k
    column number: 0 <-> k-1

    replace all elements of X[column number] X with random_sample[column_number]
    """

    # need to implement random permutation.

    if column_number >= random_sample.shape[0]:
        raise ValueError("column {} entered. Column # should be"
                         "less than {}".format(column_number, random_sample.shape[0]))

    value_chosen = perturbation_strategy(X, column_number, random_sample)
    constant_array = np.repeat(value_chosen, X.shape[0])
    X[:, column_number] = constant_array
    return X


def main():
    pass

if __name__ == '__main__':
    main()
