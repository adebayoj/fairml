from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import six


def verify_black_box_function(predict_method, number_of_features,
                              number_of_data_points=10):

    # check estimator variable is a callable.
    if not six.callable(predict_method):
        raise Exception("Please pass in a callable.")

    # now generate test data to verify that estimator is working
    covariance = np.eye(number_of_features)
    mean = np.zeros(number_of_features)

    data = np.random.multivariate_normal(mean, covariance,
                                         number_of_data_points)
    try:
        output = predict_method(data)

        # check to make sure that the estimator returns a numpy
        # array
        if type(output).__module__ != 'numpy':
            raise ValueError("Output of predict function is not "
                             "a numpy array")

        if output.shape[0] != number_of_data_points:
            raise Exception("Predict does not return an output "
                            "for every data point.")
    except:
        print("Unexpected error: ", sys.exc_info()[0])

    return True


def verify_input_data(input_dataframe):
    """
    This function assumes the black box estimator is working as required.
    Checks the dataframe to make sure there are no NANs.

    Will extend to just read a csv in the future.

    returns :- list of columns (feature names)
    """

    # first check that it is a dataframe.
    if type(input_dataframe).__module__ != 'pandas.core.frame':
        print("Input data is not a dataframe. Make sure input data is of type"
              "pandas.core.frame")
        raise

    # now we know input is a dataframe, let's check for null values
    if input_dataframe.isnull().any().sum() > 0:
        print("Input data contains some null values. Please check and handle"
              "appropriately.")
        raise

    # no input values, now check column names
    return True, list(input_dataframe.columns)


def main():
    pass


if __name__ == '__main__':
    main()
