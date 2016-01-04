
import pandas as pd

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

import pickle

import sys
import os

def sample_data_frame_return_x_y_rf_file(dataframe, contains_y, y_variable_name, num_samples):
	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		new_dataframe.drop([y_variable_name],inplace=True, axis=1)
		column_list = list(new_dataframe.columns)
		return new_dataframe.values, y, column_list
	else:
		raise "Input file to sample should always contain the y variable. "

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
	X, y, column_list_for_sampled = sample_data_frame_return_x_y_rf_file(df, True, target, num_samples)

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

	print "fitting model"
    #train new classifier on the entire dataset
	X, y, column_list_for_sampled = sample_data_frame_return_x_y_rf_file(df, True, target, num_samples=df.shape[0])

	clf_final = RandomForestRegressor(n_estimators=num_trees_final_clf, max_depth = best_params["max_depth"], min_samples_leaf = best_params["min_samples_leaf"],  min_samples_split = best_params["min_samples_split"], bootstrap = best_params["bootstrap"], max_features = best_params["max_features"])

	print "Fitting Random Forest Regressor"
	clf_final.fit(X,y)
	return clf_final, column_list_for_sampled


def obtain_feature_importance_from_rf(clf, column_names, file_path):
	feature_importance = clf.feature_importances_
	print "######################################################"
	print feature_importance
	print "######################################################"
	print column_names
	print "######################################################"
	
	if len(column_names) == len(feature_importance):
		print "feature importance == column_names"

	random_forest_combine_dict = {}
	for i in range(len(column_names)):
		print column_names[i] + " ------>>>> " + str(feature_importance[i])
		random_forest_combine_dict[column_names[i]] = feature_importance[i]

	pickle_path = file_path + "/random_forest_feature_ranking.pickle"

	with open(pickle_path, 'wb', ) as handle:
  		pickle.dump(random_forest_combine_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  	return "Pickled random forest rankings"



