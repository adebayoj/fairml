import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


from fairml import orthogonal_projection

propublica_data = pd.read_csv(filepath_or_buffer="./doc/example_notebooks/propublica_data_for_fairml.csv", sep=",",
                              header=0)

compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)


clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(propublica_data.values, compas_rating)

print(list(propublica_data.columns))

total, _ = audit_model(clf, propublica_data,
                       problem_class="classification",
                       direct_input_pertubation="constant median",
                       number_of_runs=10,
                       include_interactions=False,
                       external_data_set=None)

