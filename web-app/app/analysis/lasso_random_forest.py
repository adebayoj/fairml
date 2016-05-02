
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn import preprocessing

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

from time import time
from scipy.stats import randint as sp_randint

from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                                  LassoLarsCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.utils import ConvergenceWarning

from utils import sample_data_frame_return_x_y_column_name
from utils import scale_input_data

import pickle

import sys
import os


def hyperparameter_search_random(X, y, clf, dict_params, num_iterations):
	random_search = RandomizedSearchCV(clf, param_distributions=dict_params, n_iter=num_iterations, verbose = 2)
	start = time()
	random_search.fit(X, y)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), num_iterations))
	print "######################################################"
	print 
	print "search scores\n"
	print random_search.grid_scores_
	print ""
	print random_search.best_score_
	return random_search.best_estimator_, random_search.best_params_

def return_best_rf_regressor(df, target, num_trees_hyperparameter, num_trees_final_clf, num_iterations):
	print "entering return best rf regressor function"
	if df.shape[0] < 10000:
		num_samples = df.shape[0]
	else:
		num_samples = int(df.shape[0]*0.7)

	print "Sample dataframe"
	#use
	X, y, column_list_for_sampled = sample_data_frame_return_x_y_column_name(df, True, target, num_samples)

	# figure out a vary this some how
	"""
	param_dist = {"max_depth": [5, None],
              "max_features": sp_randint(1, df.shape[1]),
              "min_samples_split": sp_randint(1, 15),
              "min_samples_leaf": sp_randint(1, 15),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    """
	param_dist = {"max_depth": [5, None], "max_features": sp_randint(1, df.shape[1]), "min_samples_split": sp_randint(1, 15), "min_samples_leaf": sp_randint(1, 15), "bootstrap": [True]}

	clf = RandomForestRegressor(n_estimators=num_trees_hyperparameter)
	print "starting hyperparameter search"
	clf_best, best_params = hyperparameter_search_random(X, y, clf, param_dist, num_iterations)

	print "sample data for fitting model"
    #train new classifier on the entire dataset
	X, y, column_list_for_sampled = sample_data_frame_return_x_y_column_name(df, True, target, num_samples=df.shape[0])

	clf_final = RandomForestRegressor(n_estimators=num_trees_final_clf, max_depth = best_params["max_depth"], min_samples_leaf = best_params["min_samples_leaf"],  min_samples_split = best_params["min_samples_split"], bootstrap = best_params["bootstrap"], max_features = best_params["max_features"])

	print "Fitting Random Forest Regressor"
	clf_final.fit(X,y)
	return clf_final, column_list_for_sampled


def obtain_feature_importance_from_rf(clf, column_names, file_path):
	feature_importance = clf.feature_importances_

	std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
	print "######################################################"
	print feature_importance
	print "######################################################"
	print column_names
	print "######################################################"
	
	if len(column_names) == len(feature_importance):
		print "feature importance == column_names"

	random_forest_combine_dict = {}
	for i in range(len(column_names)):
		print column_names[i] + " ------>>>> " + str(feature_importance[i]), str(std[i])
		random_forest_combine_dict[column_names[i]] = (feature_importance[i], std[i])

	pickle_path = file_path + "/random_forest_feature_ranking.pickle"

	with open(pickle_path, 'wb', ) as handle:
  		pickle.dump(random_forest_combine_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  	return "Pickled random forest rankings"


def run_lasso_on_input(df, target):
   
	X_part, y_part, _ = sample_data_frame_return_x_y_column_name(df, True, target, int(0.7*df.shape[0]))

	X_part, _ = scale_input_data(X_part)

	print "#######################################"
	print "Starting LARS CV"
	print "#######################################"

	lars_cv = LassoLarsCV(cv=10).fit(X_part, y_part)

	print "#######################################"
	print "Done with LARS CV"
	print "#######################################"

	#alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
	
	X, y, column_list_for_sampled = sample_data_frame_return_x_y_column_name(df, True, target, df.shape[0])

	X, _ = scale_input_data(X)

	print "#######################################"
	print "Starting main lasso"
	print "#######################################"

	clf = RandomizedLasso(alpha= lars_cv.alphas_, random_state=12, n_resampling= 400, normalize=True).fit(X, y) 

	print "#######################################"
	print "Done with main lasso"
	print "#######################################"

	return clf, column_list_for_sampled

def obtain_feature_importance_from_lasso(clf, column_names, file_path):
	feature_importance = clf.scores_

	total_scores_for_all_features = clf.all_scores_
	mean_scores = np.mean(total_scores_for_all_features, axis = 1)
	std_scores = np.std(total_scores_for_all_features, axis=1)

	#check if the len(mean_scores and )
	if len(mean_scores) != len(std_scores):
		raise "Length of mean vector different from length of std vector for Lasso \
				This means that the vector of coefficients from Lasso was improperly handled"

	if (len(mean_scores)) != len(column_names):
		print "length of columns --->>" + str(len(mean_scores))
		print "length of feature names --->>" + str(len(column_names))
		raise "Length of feature scores returned is not the same as the length of name columns"

	complete_feature_importance_dictionary = {}
	for i in range(len(mean_scores)):
		complete_feature_importance_dictionary[column_names[i]] = (mean_scores[i], std_scores[i])

	print "########################################"
	print "Pickling Feature Importance Dictionary"
	print "########################################"


	print complete_feature_importance_dictionary

	file_path_to_pickle = file_path + "/lasso_feature_ranking.pickle"

	print "File path to write pickle file --->>>> " + file_path_to_pickle

	with open(file_path_to_pickle, 'wb') as handle2:
		pickle.dump(complete_feature_importance_dictionary, handle2, protocol=pickle.HIGHEST_PROTOCOL)

	return "Finished Pickling lasso ranking."





