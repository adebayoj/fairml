import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # issue with virtual environments and backend
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

# set plotting parameters
plt.rcParams['figure.figsize'] = (8, 4)
sns.set_style("white",
              {"axes.facecolor": "1.0",
               'font.family': [u'sans-serif'],
               'ytick.color': '0.25',
               'grid.color': '.8',
               'axes.grid': False,
               }
              )

# read in propublica data
propublica_data = pd.read_csv(
    filepath_or_buffer="./doc/example_notebooks/"
    "propublica_data_for_fairml.csv",
    sep=",",
    header=0)


# quick data processing
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)


# specify build a classifier
clf = RandomForestClassifier(n_estimators=50)

# train, test split of data
X_train, X_test, y_train, y_test = train_test_split(propublica_data,
                                                    compas_rating,
                                                    test_size=0.20,
                                                    random_state=42)

max_features = propublica_data.shape[1]

# specify parameters for hyperparameter search
# specify parameters and distributions to sample from
param_dist = {"max_depth": [6, None],
              "max_features": sp_randint(1, max_features),
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(1, 30),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# run randomized search
n_iter_search = 60
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)


# let's start training the model.
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

print(random_search.best_params_)

# now let's train a model on the entire training set
# with those parameters

clf = RandomForestClassifier(n_estimators=500, **random_search.best_params_)
clf.fit(X_train, y_train)

# now let's evaluate the classifier
probas_ = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)

# let's quickly get accuracy
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

plt.plot(fpr, tpr, lw=2, color='r',
         label='ROC (area = %0.2f) & accuracy= %0.2f' % (roc_auc,
                                                         test_accuracy))
plt.title("ROC curve on test set")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('random_forest_roc_curve.png',
            transparent=False, bbox_inches='tight')

plt.clf()


# now let's audit this model.
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
    title="FairML feature dependence for random forest model",
    save_path="fairml_random_forest.png",
    show_plot=True
)
