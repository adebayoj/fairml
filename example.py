import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

# read in propublica data
propublica_data = pd.read_csv(
    filepath_or_buffer="./doc/example_notebooks/"
    "propublica_data_for_fairml.csv",
    sep=",",
    header=0)

# quick data processing
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)

#  quick setup of Logistic regression
#  perhaps use a more crazy classifier
clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(propublica_data.values, compas_rating)

#  call audit model
importancies, _ = audit_model(
    clf.predict,
    propublica_data,
    distance_metric="mse",
    direct_input_pertubation_strategy="constant-zero",
    number_of_runs=10,
    include_interactions=False,
    external_data_set=None
)

# print feature importance
print(importancies)

# generate feature dependence plot
_ = plot_dependencies(
    importancies.get_compress_dictionary_into_key_median(),
    reverse_values=False,
    title="FairML feature dependence logistic regression model",
    save_path="fairml_propublica_linear_direct.png",
    show_plot=True
)

# let's build more complicated models.
