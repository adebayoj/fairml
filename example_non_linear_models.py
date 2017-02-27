import pandas as pd
import numpy as np
from itertools import cycle

import matplotlib
matplotlib.use('Agg')  # issue with virtual environments and backend
import matplotlib.pyplot as plt
# import seaborn as sns

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

plt.style.use('ggplot')

# set hyperparameter print verbosity for sklear
_VERBOSITY = 2
_RF_iterations = 2

# set plotting parameters
plt.rcParams['figure.figsize'] = (8, 4)

"""
sns.set_style("white",
              {"axes.facecolor": "1.0",
               'font.family': [u'sans-serif'],
               'ytick.color': '0.25',
               'grid.color': '.8',
               'axes.grid': False,
               }
              )
"""

# read in propublica data
propublica_data = pd.read_csv(
    filepath_or_buffer="./doc/example_notebooks/"
    "propublica_data_for_fairml.csv",
    sep=",",
    header=0)


# quick data processing
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)

X = StandardScaler().fit_transform(propublica_data.values)

# train, test split of data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    compas_rating,
                                                    test_size=0.20,
                                                    random_state=42)

###########################

# Hyper-parameter Search for random forest
# Takes too long to do this for all models.

##########################

# specify build a classifier
clf = RandomForestClassifier(n_estimators=50)

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
n_iter_search = _RF_iterations
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   verbose=_VERBOSITY)


# let's start training the model.
print("Beginning hyper-parameter search")
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

print(random_search.best_params_)

# now let's train a model on the entire training set
# with those parameters


# 'gp': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),

###########################

# Setup classifiers to test
#  'GP': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
##########################
classifiers_dict = {'Random_Forest': RandomForestClassifier(
    n_estimators=100,
    **random_search.best_params_),
    'Logit': LogisticRegression(penalty='l2', C=0.01),
    'Neural_Network': MLPClassifier(hidden_layer_sizes=(100, 3),
                                    max_iter=150,
                                    alpha=1e-4,
                                    solver='adam',
                                    verbose=10,
                                    tol=1e-5,
                                    random_state=1,
                                    learning_rate_init=.1),
    'SVM_Linear': SVC(kernel="linear", C=0.025, probability=True),
    'GP': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
}


# colors for plots
colors = cycle(['cyan', 'red', 'black', 'magenta', 'green'])

for dict_key, color in zip(classifiers_dict.items(), colors):
    key = dict_key[0]
    clf = dict_key[1]
    print("Fitting {} classifier".format(key))
    clf = classifiers_dict[key]
    clf.fit(X_train, y_train)
    classifiers_dict[key] = clf
    print("Done Fitting {} classifier".format(key))

    print("Checking model performance on test set.")
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # let's quickly get accuracy
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    plt.plot(fpr,
             tpr,
             lw=2,
             color=color,
             label='Model: %s, '
             'ROC (area = %0.2f) & accuracy= %0.2f' % (key,
                                                       roc_auc,
                                                       test_accuracy))
plt.title("ROC curve on test set for models")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_curve_all_models.png',
            transparent=False, bbox_inches='tight')
plt.clf()


###########################

# Now let's audit each model.

##########################
for key in classifiers_dict:

    print("auditing model {}".format(key))

    importancies, _ = audit_model(
        classifiers_dict[key].predict,
        propublica_data,
        distance_metric="mse",
        direct_input_pertubation_strategy="constant-zero",
        number_of_runs=10,
        include_interactions=False,
        external_data_set=None
    )

    # generate feature dependence plot
    _ = plot_dependencies(
        importancies.get_compress_dictionary_into_key_median(),
        reverse_values=False,
        title="FairML feature dependence for {} model".format(key),
        save_path="{}_feature_dependence_model.png".format(key),
        show_plot=True
    )

    plt.clf()
