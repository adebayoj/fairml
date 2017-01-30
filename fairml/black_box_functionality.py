import numpy as np
import pandas as pd
import sys


def verify_black_box_estimator(estimator, number_of_features,
                               number_of_data_points=10):

    # first check that estimator object has predict method
    try:
        predict_method = estimator.predict
    except AttributeError:
        print("Input estimator does not have a predict method")
        raise
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    # now generate test data to verify that estimator is working
    covariance = np.eye(number_of_features)
    mean = np.zeros(number_of_features)

    data = np.random.multivariate_normal(mean, covariance,
                                         number_of_data_points)

    try:
        output = estimator.predict(data)

        # check to make sure that the estimator returns a numpy
        # array
        if type(output).__module__ != 'numpy':
            print(
                "Output of estimator's predict is not a numpy array")

            raise

        if output.shape[0] != number_of_data_points:
            print("Predict does not return an output for every data point.")
            raise
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

    #no input values, now check column names
    return True, list(input_dataframe.columns)


"""
class BlackBoxModel(object):

    def __init__(self, estimator, number_of_features,
                 Regressor=True, Classifier=False):
        pass
        # check that the estimator is callable.

    def make_predictions(self, numpy_array_of_input):

        output = estimator.predict(numpy_array_of_input)

        return output

"""


def main():
    pass

if __name__ == '__main__':
    main()
