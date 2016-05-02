import os
import sys
import argparse
from argparse import ArgumentParser
import pandas as pd
import time
from subprocess import call
import time

import numpy as np


from mrmr_wrapper import call_mrmr_routine
from mrmr_wrapper import remove_mrmr_input_folder_to_clean_up_space
from lasso_random_forest import obtain_feature_importance_from_rf
from lasso_random_forest import return_best_rf_regressor
from lasso_random_forest import run_lasso_on_input
from lasso_random_forest import obtain_feature_importance_from_lasso

from clean_up_mrmr_output import aggregate_mrmr_results_and_pickle_dictionary
from clean_up_mrmr_output import write_out_rankings
from clean_up_mrmr_output import get_list_of_files
from clean_up_mrmr_output import convert_to_float


from utils import sample_data_frame_return_x_y_column_name
from utils import scale_input_data
from utils import pickle_this_variable_with_this_name_to_this_folder

#sklearn imports
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
from sklearn import cross_validation
from sklearn.linear_model import LassoCV

from numpy.linalg import inv



"""
function: get_parallel_vector
Input: two numpy (v1, v2) arrays (vectors) of the same length. 
Output: returns a vector parallel_v2, which is the component of 
		v2 that is parallel to v1
"""

def get_parallel_vector(v1, v2):
    
    #check that the two vectors are the same length
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if v1.shape[0] != v2.shape[0]:
        return "Error, both vectors are not of the same length"
    
    scaling = np.dot(v1,v2)/np.dot(v1, v1)
    parallel_v2 = (scaling * v1)
    
    return parallel_v2

'''
This function takes a vector v1 & v2 as arguments, and then returns vector 

v3 which is the component of v2 that is orthogonal(perpendicular) to v1

v1 = reference vector (numpy array)
v2 = a vector whose orthogonal component you want to get (numpy array)

'''
def get_orthogonal_vector(v1, v2):
    
    #check that the two vectors are the same length
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if v1.shape[0] != v2.shape[0]:
        return "Error, both vectors are not of the same length"
    
    scaling = np.dot(v1,v2)/np.dot(v1, v1)
    orthogonal_v2 = v2 - (scaling * v1)
    
    return orthogonal_v2

"""
Takes in two arguments, a numpy two dimensional array, and a numpy vector. 
computes the projection matrix for A (the two dimensional array), and 
returns the component of the vector that is orthogonal to the subspace spanned by the
columns of A. 
"""

def orthogonal_vector_to_subspace(A, v1):

	#print "performing subspace orthogonal calculation"
    
	#print "computing inverse"
	m = inv(np.dot(A.T, A))
    
	#print "computing projection"
	projection = np.dot(np.dot(A, m), A.T)
    
	orthogonal_projection = np.eye(len(v1)) - projection
    
	orthogonal_v1 = np.dot(orthogonal_projection, v1)
    
	#print "returning orthogonal vector"

	return orthogonal_v1

"""
Takes in a numpy array v1, and then returns a numpy 2d array, A,  where the columns of 
A correspond to different non linear transformations applied to v1

Current transformations
Polynomial: i.e x^2, x^3 etc
Log: log(x)
square root: x^(1/2)
exponential: e^x
sin: sin(x)
cos: cos(x)

"""
def return_non_linear_transformation(v1, poly, log, square_root, exponential, sin, cos):
    A = np.zeros((v1.shape[0], 1))
    A[:,0] = v1
    #print A.shape
    if poly > 1:
        for i in range(2,poly+1):
            current_power = v1**i
            current_power = np.reshape(current_power, (current_power.shape[0], 1))
            A = np.append(A, current_power, axis = 1)
            
    if square_root:
        sqrt= np.sqrt(u3 + np.abs(min(u3)) + 1)
        sqrt = np.reshape(sqrt, (sqrt.shape[0], 1))
        A = np.append(A, sqrt, axis = 1)
        
    if exponential:
        exp_term = np.exp(v1 - np.max(v1))
        exp_term = np.reshape(exp_term, (exp_term.shape[0], 1))
        A = np.append(A, exp_term, axis=1)
        
    if log:
        #shift the entire vector by the mininum value is +1
        log_term = np.log(v1 + np.abs(min(v1)) + 1)
        log_term = np.reshape(log_term, (log_term.shape[0], 1))
        A = np.append(A, log_term, axis=1)
        
    if sin:
        sin_term = np.sin(v1)
        sin_term = np.reshape(sin_term, (sin_term.shape[0], 1))
        A = np.append(A, sin_term, axis=1)
        
    if cos:
        cos_term = np.cos(v1)
        cos_term = np.reshape(cos_term, (cos_term.shape[0], 1))
        A = np.append(A, cos_term, axis=1)
    
    #print A.shape
    return A

def perform_orthogonal_variable_selection(X, y, column_list_for_sampled, non_linear, feature_dictionary):

	X, _ = scale_input_data(X)

	#train base line model with the original model
	#define lasso model

	print "defining lasso model "
	clf_baseline = LassoCV(n_alphas=3, cv=3)

	print "fitting lasso model with cross validation library"
	base_line_lasso = cross_validation.cross_val_score(clf_baseline, X, y, cv = 3, n_jobs=1)

	print "done fitting lasso library, now computing baseline performance"

	#baseline performance 
	baseline_performance = np.mean(base_line_lasso)	

	print baseline_performance

	print "baseline performance printed above"

	#get number of columns and rows for main data
	rows, columns = X.shape 


	print "starting iterative orthogonal transformation for each feature"

	for i in range(columns):
		feature_name_current_i = column_list_for_sampled[i]
		print "Currently on ----->> " + feature_name_current_i
		current_column = X[:,i] 
			
		transformed_X = np.zeros((rows, columns))
		
		if non_linear > 0:
			#print "performing non-linear transformation"
			A = return_non_linear_transformation(current_column, poly=3, log=True,square_root=False, exponential=False, sin=False, cos=False)
			for j in range(columns):
				#print "in inner loop for " + feature_name_current_i  + " ---->> " + column_list_for_sampled[j]
				if i==j:
					transformed_X[:, j]  = np.zeros((rows))
				else:
					transformed_X[:, j] = orthogonal_vector_to_subspace(A, X[:,j])

		else:
			#print "performing linear transformation"
			for j in range(columns):
				#print "in inner loop for " + feature_name_current_i  + " ---->> " + column_list_for_sampled[j]
				if i==j:
					#skip the current columns, because we are removing it from the
					#analysis. 
					transformed_X[:, j]  = np.zeros((rows))

				else:
					transformed_X[:, j] = get_orthogonal_vector(current_column, X[:,j])

		#print "done with computing transformed matrix"
		#print "original matrix size"
		#print X.shape
		#print "transformed matrix size"
		#print transformed_X.shape


		#now we have transformed_x the same size as original X with the column i having all zeros. 
		# since we have our own new classifier then delete original columne
		transformed_X = np.delete(transformed_X, i, 1)

		#print "deleted original column ----->> " + feature_name_current_i

		#print "transformed matrix size after deleting original columns"
		#print transformed_X.shape

		#print "training and fitting lasso model on transformed data set"
		#now we have have a transformed vector with which we are ready to do another prediction. 
		lasso_scores_i = cross_validation.cross_val_score(clf_baseline, transformed_X, y, cv = 3, n_jobs=1)

		#print "computing scores for the model"
		current_feature_score = np.mean(lasso_scores_i)

		#print current_feature_score
		#print "score for current feature shown above"
		change_in_score = baseline_performance - current_feature_score

		feature_dictionary[feature_name_current_i].append(change_in_score)

	return feature_dictionary



def orthogonal_variable_selection_cannot_query_black_box(df, target, non_linear, no_bootstrap_iter, num_samples):

	master_dictionary = {}
	for name in list(df.columns):
		if name != target:
			master_dictionary[name] = []

	for iteration in range(no_bootstrap_iter):

		print "going through iteration " + str(iteration) + " of orthogonal feature selection"

		X, y, column_list_for_sampled = sample_data_frame_return_x_y_column_name(df, True, target, num_samples)

		master_dictionary = perform_orthogonal_variable_selection(X, y, column_list_for_sampled, non_linear, master_dictionary)

	return master_dictionary

def aggregate_orthogonal_rankings(input_dictionary, ranking_path):

	file_name = "orthogonal_projection_feature_ranking.pickle"
	file_path = ranking_path + '/' + file_name

	final_output_dictionary = {}
	for feature in input_dictionary:
		scores = input_dictionary[feature]
		final_output_dictionary[feature] = (np.mean(np.array(scores)) , np.std(np.array(scores)))

	return pickle_this_variable_with_this_name_to_this_folder(final_output_dictionary, file_path)






