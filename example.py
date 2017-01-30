import pandas as pd
from sklearn.linear_model import LogisticRegression

#import specific projection format. 
from fairml import orthogonal_projection

#read in propublica data
propublica_data = pd.read_csv(filepath_or_buffer="./doc/example_notebooks/propublica_data_for_fairml.csv", sep=",",
                              header=0)

#quick processing
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)


#quick setup of Logistic regression
#perhaps use a more crazy classifier
clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(propublica_data.values, compas_rating)

#double check list of input data columns
print(list(propublica_data.columns))

#call 
total, _ = audit_model(clf, propublica_data,
                       problem_class="classification",
                       direct_input_pertubation="constant median",
                       number_of_runs=10,
                       include_interactions=False,
                       external_data_set=None)

