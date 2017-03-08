import matplotlib

# temporary work around down to virtualenv
# matplotlib issue.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.linear_model import LogisticRegression

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 8)

# read in propublica data
propublica_data = pd.read_csv("./doc/example_notebooks/"
                              "propublica_data_for_fairml.csv")

# quick data processing
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)

#  quick setup of Logistic regression
#  perhaps use a more crazy classifier
clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(propublica_data.values, compas_rating)

#  call audit model
importancies, _ = audit_model(clf.predict, propublica_data)

# print feature importance
print(importancies)

# generate feature dependence plot
fig = plot_dependencies(
    importancies.median(),
    reverse_values=False,
    title="FairML feature dependence logistic regression model"
)

file_name = "fairml_propublica_linear_direct.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)
