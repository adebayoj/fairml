
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
                                  LassoLarsCV, LassoCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.utils import ConvergenceWarning

import pickle

import sys
import os

def scale_input_data(X):
	 scaler = StandardScaler()
	 scaler.fit(X)
	 X = scaler.transform(X.copy())
	 return X, scaler


def sample_data_frame_return_x_y_rf_file(dataframe, contains_y, y_variable_name, num_samples):
	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		new_dataframe.drop([y_variable_name],inplace=True, axis=1)
		column_list = list(new_dataframe.columns)
		return new_dataframe.values, y, column_list
	else:
		print "whatever"

def run_lasso_on_input(df, target):
   
	X_part, y_part, _ = sample_data_frame_return_x_y_rf_file(df, True, target, int(0.7*df.shape[0]))

	X_part, _ = scale_input_data(X_part)

	print "#######################################"
	print "Starting LARS CV"
	print "#######################################"

	lars_cv = LassoLarsCV(cv=10).fit(X_part, y_part)

	print "#######################################"
	print "Done with LARS CV"
	print "#######################################"

	print lars_cv.alphas_

	print "just printed lars_cv alphas"

	alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 20)
	
	#print alphas

	print "Just printed the alphas found from lars_cv"

	X, y, column_list_for_sampled = sample_data_frame_return_x_y_rf_file(df, True, target, df.shape[0])

	X, _ = scale_input_data(X)

	print "fiting randomized lasso to the entire input data"

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
	print "Feature Importance Dictionary"
	print "########################################"

	return complete_feature_importance_dictionary



full_input_data = pd.read_csv(filepath_or_buffer='../data/processed_data_sets/turkey_credit_individual_data_with_pd_limit.csv', sep=',')
full_input_data.dropna(axis=0, how='any', inplace=True)

clf, columns = run_lasso_on_input(full_input_data, 'credit_limit')

f = obtain_feature_importance_from_lasso(clf, columns, "whatever")

pickle_path = "lasso.pickle"

with open(pickle_path, 'wb', ) as handle:
  	pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


