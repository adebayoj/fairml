import numpy as np
import pandas as pd
from collections import defaultdict
from random import randint
import six

# import a few utility functions
from .utils import mse
from .utils import accuracy
from .utils import replace_column_of_matrix
from .utils import constant_median

# black_box_functionality has functions to verify inputs.
from .black_box_functionality import verify_black_box_estimator
from .black_box_functionality import verify_input_data


class AuditResult(dict):
    def __repr__(self):
        output = []
        for key, value in self.items():
            importance = np.median(np.array(value))
            output.append("Feature: {},\t Importance: {}"
                          .format(key, importance))
        return "\n".join(output)


def get_parallel_vector(v1, v2):
    """
    Parameters
    ------------

    v1 - baseline vector (numpy)
    v2 - vector that you'd like to make parrallel to v1

    Returns
    -------------
    parallel_v2, projection of v2 that is parallel to v1

    """

    # check that the two vectors are the same length
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.shape[0] != v2.shape[0]:
        return "Error, both vectors are not of the same length"

    scaling = np.dot(v1, v2)/np.dot(v1, v1)
    parallel_v2 = (scaling * v1)
    return parallel_v2


def get_orthogonal_vector(v1, v2):
    """
    Parameters
    ------------

    v1 - baseline vector (numpy)
    v2 - vector that you'd like to make orthogonal to v1

    Returns
    -------------
    orthogonal_v2, projection of v2 that is orthogonal to v1

    """

    # check that the two vectors are the same length
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.shape[0] != v2.shape[0]:
        return "Error, both vectors are not of the same length"

    scaling = np.dot(v1, v2)/np.dot(v1, v1)
    orthogonal_v2 = v2 - (scaling * v1)
    return orthogonal_v2


def obtain_orthogonal_transformed_matrix(X, baseline_vector,
                                         column_to_skip=-1):
    """
    X is the column that has the data

    orthogonal vector is a baseline vector that we want to make the columns of
    X orthogonal to.

    skip column_to_skip if possible.
    """

    # first check to make sure that the matrix and vector have similar lengths
    # for shape
    if X.shape[0] != baseline_vector.shape[0]:
        raise ValueError('Need to be the same shape')

    for column in range(X.shape[1]):
        # you might want to skip the constant column
        # for interactions, you don't actually have them in the data
        # so you don't want to skip any column.
        if column == column_to_skip:
            continue
        orthogonal_column = get_orthogonal_vector(
            baseline_vector, X[:, column])
        X[:, column] = orthogonal_column
    return X


def audit_model(predict, input_dataframe, distance_metric="mse",
                direct_input_pertubation="constant median",
                number_of_runs=5, include_interactions=False,
                external_data_set=None):
    """
    Estimator -> Black-box function that has a predict method

    input_dataframe -> dataframe with shape (n_samples, n_features)

    distance_metric -> one of ["mse", "accuracy"], this
                variable defaults to regression.

    direct_input_pertubation -> This is referring to how to zero out a
                            single variable. One of three different options
                            1) replace with a random constant value
                            2) replace with median constant value
                            3) replace all values with a random permutation of
                               the column.  options = [constant-zero,
                               constant-median, global-permutation]

    number_of_runs -> number of runs to perform.

    external_data_set ->data that did not go into training the model, but
                        that you'd like to see what impact that data
                        has on the black box model.

                        (VERY IMPORTANT if enabled.)
                        You need to make sure that number of rows in this
                        dataframe matches that of the input data. This is
                        because we'll be using the input_dataframe as a
                        foundational dataset and making the columns of that
                        matrix orthogonal to each of the different columns in
                        this data frame to check their influence.


    """
    assert isinstance(input_dataframe, pd.DataFrame), ("Data must be a pandas "
                                                       "dataframe")
    assert distance_metric in ["mse", "accuracy"], ("Distance metric must be "
                                                    "'mse' or 'accuracy'")

    if not isinstance(direct_input_pertubation, six.callable):
        try:
            direct_input_pertubation = some_dictionary[direct_input_pertubation]
        except KeyError:
            raise Exception("Invalid selection for direct_input_pertubation. "
                            "Must be callable or one of {}".format(", ".join(some_dictionary.keys())))

    # create output dictionaries
    direct_pertubation_feature_output_dictionary = defaultdict(list)
    complete_perturbation_dictionary = defaultdict(list)

    # check if estimator has predict function
    # if check then test estimator for prediction and numpy variable return.
    # It'll raise errors if there are issues with passed in estimator.
    number_of_features = input_dataframe.shape[1]

    _ = verify_black_box_estimator(estimator, number_of_features)

    # verify data set and black_box editor.
    _, list_of_column_names = verify_input_data(input_dataframe)

    # convert data to numpy array
    data = input_dataframe.values

    # get the normal output
    normal_black_box_output = estimator.predict(data)

    # perform the straight forward linear search at first
    for current_iteration in range(number_of_runs):
        random_row_to_select = randint(0, data.shape[0]-1)
        random_sample_selected = data[random_row_to_select, :]

        # go over every column
        for col in range(number_of_features):
            # get reference vector
            reference_vector = data[:, col]
            data_col_ptb = replace_column_of_matrix(
                np.copy(data),
                col,
                random_sample_selected,
                perturbation_strategy="constant-zero")
            output_constant_col = estimator.predict(data_col_ptb)
            if distance_metric == "accuracy":
                output_difference_col = accuracy(
                    output_constant_col, normal_black_box_output)
            else:
                output_difference_col = mse(
                    output_constant_col, normal_black_box_output)

            # store independent output by themselves
            direct_pertubation_feature_output_dictionary[
                list_of_column_names[col]].append(output_difference_col)

            # now make all the remaining columns of the matrix $data_copy_with_constant_column$
            # except $col$ orthogonal to current vector of interest.

            total_ptb_data = obtain_orthogonal_transformed_matrix(
                data_col_ptb,
                reference_vector,
                column_to_skip=col)

            total_transformed_output = estimator.predict(total_ptb_data)

            if distance_metric == "accuracy":
                total_difference = accuracy(
                    total_transformed_output, normal_black_box_output)
            else:
                total_difference = mse(
                    total_transformed_output, normal_black_box_output)

            complete_perturbation_dictionary[
                list_of_column_names[col]].append(total_difference)
            
    return (AuditResult(complete_perturbation_dictionary),
            AuditResult(direct_pertubation_feature_output_dictionary))


def main():
    pass

if __name__ == '__main__':
    main()
