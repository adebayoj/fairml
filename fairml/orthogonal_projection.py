import os

"""
https://stackoverflow.com/questions/8085520/generating-pdf-latex-with-python-script
"""

def audit_model(estimator, sample_dataframe, classifier=False,
                output_name_present_in_dataframe=None):

    #check if estimator has predict function
    number_of_features = sample_dataframe.shape[1]

    #if check then test estimator for prediction and numpy variable return.
    _ = verify_black_box_estimator(estimator, number_of_features)

    # verify data set and black_box editor. 

    #perform the straight forward linear search at fist




    pass
