import os

from utils import mse
from utils import accuracy


from black_box_funtionality import verify_black_box_estimator
from black_box_funtionality import verify_input_data

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
    
    #check that the two vectors are the same length
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if v1.shape[0] != v2.shape[0]:
        return "Error, both vectors are not of the same length"
    
    scaling = np.dot(v1,v2)/np.dot(v1, v1)
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


def audit_model(estimator, input_dataframe, problem_class="regression",
                direct_input_pertubation="constant median",
                number_of_runs=10, include_interactions=False, 
                external_data_set=None):
    """
    Estimator -> Black-box function that has a predict method

    input_dataframe -> dataframe with shape (n_samples, n_features)

    problem_class -> one of ["regression", "classification], this 
                variable defaults to regression. 

    direct_input_pertubation -> This is referring to how to zero out a 
                            single variable. One of three different options
                            1) replace with a random constant value
                            2) replace with median constant value
                            3) replace all values with a random permutation of the column. 

    number_of_runs -> number of runs to perform. 

    external_data_set -> data that did not go into training the model, but
                        that you'd like to see what impact that data
                        has on the black box model. 


    """

    # check if estimator has predict function
    number_of_features = sample_dataframe.shape[1]

    # if check then test estimator for prediction and numpy variable return.
    # It'll raise errors if there are issues with passed in estimator.
    _ = verify_black_box_estimator(estimator, number_of_features)

    # verify data set and black_box editor.
    _, list_of_column_names = verify_input_data(input_dataframe)

    # perform the straight forward linear search at first

    # if interactions are set, then perform interactions and do as well. 
    # use the actual data here and not the 


    pass
