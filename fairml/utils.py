import numpy as np
import sys


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

def main():
    pass

if __name__ == '__main__':
    main()