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

    pass


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
