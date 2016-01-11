
from utils import *
import pickle
import os
import numpy as np
from collections import defaultdict
import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams


import pandas as pd
import csv
from math import sqrt

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

from matplotlib import rc

import random

from matplotlib.font_manager import FontProperties

import seaborn as sns
import time

sns.color_palette("bright")
sns.set(font_scale=1.2)
sns.plotting_context(context="talk", rc=None)

font = {'family': 'Serif',
        'color':  'Black',
        'weight': 'normal',
        'size': 13,
        }
title_font = {'family': 'Serif',
        'color':  'Black',
        'weight': 'semibold',
        'size': 15,
        }

def organize_mrmr_ranking(pickle_file_path, folder_path, write_plot_path, figure_size, color_code, sensitive_features, methodology, figure_name, font, Title_font, target):

	final_feature_importance_dictionary = {}

	file0 = open(pickle_file_path, 'rb')
	feature_scores = pickle.load(file0)

	updated_feature_dictionary = {}
	if methodology.lower() == 'mrmr':
		mapping_file = "mrmr_column_feature_name_mapping.pickle"
		paths = folder_path + '/' + mapping_file

		file2 = open(paths, "rb")
		mrmr_ranking_names = pickle.load(file2)

		for key in mrmr_ranking_names:
			new_key = mrmr_ranking_names[key]
			updated_feature_dictionary[new_key] = feature_scores[key]
	else:
		updated_feature_dictionary = feature_scores

	feature_names  = updated_feature_dictionary.keys()
	feature_values = updated_feature_dictionary.values()

	feature_importance = []
	for tp in feature_values:
		feature_importance.append(tp[0])

	

	feature_importance = 100.0 * (np.array(feature_importance) / np.array(feature_importance).max())
	print feature_importance

	sorted_idx = np.argsort(feature_importance)

	final_column_list = []
	truncated_feature_importance = []
	for i in xrange(len(feature_importance)):
		cur_feature_name = feature_names[sorted_idx[i]]
		final_feature_importance_dictionary[cur_feature_name] = feature_importance[sorted_idx[i]]
		#cur_feature_name = " ".join(cur_feature_name.split("_"))
		cur_feature_name =  cur_feature_name[0].upper() + cur_feature_name[1:]
		final_column_list.append(cur_feature_name)
		truncated_feature_importance.append(feature_importance[sorted_idx[i]])

	rcParams['figure.figsize'] = figure_size[0], figure_size[1]

	y_pos = np.arange(len(final_column_list))+ 0.5

	color_format = []
	for i in range(len(final_column_list)):
		c_name = final_column_list[i].lower()
		if c_name in sensitive_features:
			print feature_names[i]
			color_format.append('grey')
		else:
			color_format.append(color_code)

	#print feature_names
	#print color_format

	plt.barh(y_pos, truncated_feature_importance, align='center', color = color_format)
	plt.yticks(y_pos, final_column_list, **font)
	plt.xlabel('Normalized Attribute Ranking', fontdict=font)
	plt.title('Attribute Ranking using the {0} Methodology'.format(methodology), fontdict=Title_font)

	min_x = np.min(np.array(truncated_feature_importance))

	
	plt.xlim([min_x - 2,105])
	#axes = plt.gca()

	#axes.set_xlim([min_x - 2,105])
	plt.savefig(write_plot_path + "/" + "{0}".format(figure_name), bbox_inches = 'tight')
	plt.clf()

		#go through and delete particular key value if present
	if target in final_feature_importance_dictionary.keys():
		print "yesssssssssssssss"
		del final_feature_importance_dictionary[target]

	return final_feature_importance_dictionary

#organize_mrmr_ranking(path, figure_size, color_code, sensitive_features, methodology, figure_name, font)
#always enter sensitive attribute names in lower case


def normalize_np_array(array_to_normalize):
	print array_to_normalize
	max_value, min_value = np.max(np.array(array_to_normalize)), np.min(np.array(array_to_normalize))
	range_values = max_value - min_value

	print range_values
	print min_value
	print max_value
	normalized_value = (np.array(array_to_normalize) - min_value)/ float(range_values)

	print "print normalized values"


	print normalized_value

	return np.multiply(normalized_value, 100.0)

def bootstrap_sampling_to_compute_mean_variance(array_to_use, attribute, folder_path):

	now = time.time()
	
	booststrap_array = np.random.choice(array_to_use, 1000, replace=True)
	mean = np.mean(booststrap_array)
	std = np.std(booststrap_array)

	plt.hist(np.array(booststrap_array), bins = 10)
	plt.xlabel("Quantitative Predictive Dependence")
	plt.ylabel("Counts")
	plt.title("Histogram of Predictive Dependence for {0} \n Across all 4 Methodologies".format(attribute))
	filename = str(now)
	plt.savefig(folder_path + "/" +filename + ".pdf", bbox_inches = 'tight')

	plt.clf()

	return mean, std

def combine_rankings(lasso_dictionary, mrmr_dictionary, random_forest_dictionary, orthogonal_dictionary, folder_path):

	#check the length of these dictionaries
	size_dict = set()
	size_dict.add(len(lasso_dictionary))
	size_dict.add(len(mrmr_dictionary))
	size_dict.add(len(random_forest_dictionary))
	size_dict.add(len(orthogonal_dictionary))

	if len(size_dict) > 1:
		print "Compiled feature dictionaries are not all the same size"
		print mrmr_dictionary
		print "length mrmr --->> " + str(len(mrmr_dictionary))
		print "length random forest --->> " + str(len(random_forest_dictionary))
		print "length lasso --->> " + str(len(lasso_dictionary))
		print "length orthogonal  --->> " + str(len(orthogonal_dictionary))
		raise "Compiled feature dictionaries are not all the same size"


	feature_names = lasso_dictionary.keys()
	lasso_feature_scores = normalize_np_array(lasso_dictionary.values())

	rf_feature_scores = normalize_np_array(random_forest_dictionary.values())

	orthogonal_feature_scores = normalize_np_array(orthogonal_dictionary.values())

	mrmr_feature_scores = normalize_np_array(mrmr_dictionary.values())

	master_dictionary = {}
	for name in feature_names:
		master_dictionary[name] = []

	for i in range(len(feature_names)):
		lasso_feature_name = lasso_dictionary.keys()[i]
		mrmr_feature_name = mrmr_dictionary.keys()[i]
		rf_feature_name = random_forest_dictionary.keys()[i]
		orthogonal_feature_name = orthogonal_dictionary.keys()[i]


		master_dictionary[lasso_feature_name].append(lasso_feature_scores[i])
		master_dictionary[mrmr_feature_name].append(mrmr_feature_scores[i])
		master_dictionary[rf_feature_name].append(rf_feature_scores[i])
		master_dictionary[orthogonal_feature_name].append(orthogonal_feature_scores[i])

	master_feature_dictionary_with_mean_std = {}
	for fn in master_dictionary:
		master_feature_dictionary_with_mean_std[fn] = (bootstrap_sampling_to_compute_mean_variance(master_dictionary[fn], fn, folder_path))
		#np.mean(np.array(master_dictionary[fn])), np.std(np.array(master_dictionary[fn]))

	return master_feature_dictionary_with_mean_std
	#normalize rankings btw 0 - 1

def graph_combine_plot(combined_dictionary, folder_path, figure_size, color_code, sensitive_features, figure_name, font, title_font, target):



	feature_names  = combined_dictionary.keys()
	feature_values = combined_dictionary.values()

	feature_error_values = []
	feature_importance = []
	for tp in feature_values:
		feature_importance.append(tp[0])
		feature_error_values.append(tp[1])


	#feature_importance = 100.0 * (np.array(feature_importance) / np.array(feature_importance).max())
	#print feature_importance

	sorted_idx = np.argsort(feature_importance)

	final_column_list = []
	truncated_feature_importance = []
	final_error_values = []
	for i in xrange(len(feature_importance)):
		cur_feature_name = feature_names[sorted_idx[i]]
		cur_feature_name =  cur_feature_name[0].upper() + cur_feature_name[1:]
		final_column_list.append(cur_feature_name)
		truncated_feature_importance.append(feature_importance[sorted_idx[i]])
		final_error_values.append(feature_error_values[sorted_idx[i]])

	rcParams['figure.figsize'] = figure_size[0], figure_size[1]

	y_pos = np.arange(len(final_column_list))+ 0.5

	color_format = []
	for i in range(len(final_column_list)):
		c_name = final_column_list[i].lower()
		if c_name in sensitive_features:
			print feature_names[i]
			color_format.append('grey')
		else:
			color_format.append(color_code)

	#plt.barh(y_pos, truncated_feature_importance, align='center', color = color_format, xerr=final_error_values, ecolor='k')
	plt.barh(y_pos, truncated_feature_importance, align='center', color = color_format)
	plt.yticks(y_pos, final_column_list, **font)
	plt.xlabel('Combined Attribute Ranking', fontdict=font)
	plt.title('Combined Feature Importance \n across all Methodologies', fontdict=title_font)

	max_x = np.max(np.array(truncated_feature_importance)) + np.max(final_error_values)

	
	plt.xlim([-1,max_x])
	#axes = plt.gca()

	#axes.set_xlim([min_x - 2,105])
	plt.savefig(folder_path + "/" + "{0}.pdf".format(figure_name), bbox_inches = 'tight')
	plt.clf()


	return "plotted"

'''
lasso_dictionary = organize_mrmr_ranking("fairml_analysis_2016-01-10_19:23:55/ranking_results/lasso_feature_ranking.pickle", "fairml_analysis_2016-01-10_19:23:55/ranking_results",
					"fairml_analysis_2016-01-10_19:23:55/plots", (10,10), "m", ["male", "female"], "LASSO", "LASSO_credit_limit_testing.pdf", font, title_font, "credit_limit")

mrmr_dictionary = organize_mrmr_ranking("fairml_analysis_2016-01-10_19:23:55/ranking_results/mrmr_feature_ranking.pickle", "fairml_analysis_2016-01-10_19:23:55/ranking_results",
					"fairml_analysis_2016-01-10_19:23:55/plots", (10,10), "y", ["male", "female"], "MRMR", "MRMR_credit_limit_testing.pdf", font, title_font, "credit_limit")

random_forest_dictionary = organize_mrmr_ranking("fairml_analysis_2016-01-10_19:23:55/ranking_results/random_forest_feature_ranking.pickle", "fairml_analysis_2016-01-10_19:23:55/ranking_results",
					"fairml_analysis_2016-01-10_19:23:55/plots", (10,10), "c", ["male", "female"], "Random Forest", "Random_Forest_credit_limit_testing.pdf", font, title_font, "credit_limit")

orthogonal_dictionary = organize_mrmr_ranking("fairml_analysis_2016-01-10_19:23:55/ranking_results/orthogonal_projection_feature_ranking.pickle", "fairml_analysis_2016-01-10_19:23:55/ranking_results",
					"fairml_analysis_2016-01-10_19:23:55/plots", (10,10), "r", ["male", "female"], "Orthogonal Transformation", "Orthogonal_credit_limit_testing.pdf", font, title_font, "credit_limit")



combined_full_dict =  combine_rankings(lasso_dictionary, mrmr_dictionary, random_forest_dictionary, orthogonal_dictionary, "fairml_analysis_2016-01-10_19:23:55/plots",)

graph_combine_plot(combined_full_dict, "fairml_analysis_2016-01-10_19:23:55/plots", (10,10), "cornflowerblue", ["male", "female"], "Combined_Ranking_Turkey_Credit_limit", font, title_font, "credit_limit")
'''

def run_graphing_module(path_to_feature_rankings, path_to_plot_folder, figure_size, sensitive_attributes, target, font, title_font):

	lasso_pickle_path = "lasso_feature_ranking.pickle"
	lasso_pickle_path = path_to_feature_rankings + "/" + lasso_pickle_path
	lasso_file_name = "LASSO_"+target+"_ranking_graph.pdf"

	lasso_dictionary = organize_mrmr_ranking(lasso_pickle_path, path_to_feature_rankings, path_to_plot_folder, figure_size, \
						 "m", sensitive_attributes, "LASSO", lasso_file_name, font, title_font, target)

	mrmr_pickle_path = "mrmr_feature_ranking.pickle"
	mrmr_pickle_path = path_to_feature_rankings + "/" + mrmr_pickle_path
	mrmr_file_name = "MRMR_"+target+"_ranking_graph.pdf"

	mrmr_dictionary = organize_mrmr_ranking(mrmr_pickle_path, path_to_feature_rankings, path_to_plot_folder, figure_size, \
						 "y", sensitive_attributes, "MRMR", mrmr_file_name, font, title_font, target)


	orthogonal_pickle_path = "orthogonal_projection_feature_ranking.pickle"
	orthogonal_pickle_path = path_to_feature_rankings + "/" + orthogonal_pickle_path
	orthogonal_file_name = "Orthogonal_"+target+"_ranking_graph.pdf"

	orthogonal_dictionary = organize_mrmr_ranking(orthogonal_pickle_path, path_to_feature_rankings, path_to_plot_folder, figure_size, \
						 "r", sensitive_attributes, "Orthogonal", orthogonal_file_name, font, title_font, target)

	rf_pickle_path = "random_forest_feature_ranking.pickle"
	rf_pickle_path = path_to_feature_rankings + "/" + rf_pickle_path
	rf_file_name = "Random_Forest_"+target+"_ranking_graph.pdf"

	random_forest_dictionary = organize_mrmr_ranking(rf_pickle_path, path_to_feature_rankings, path_to_plot_folder, figure_size, \
						 "c", sensitive_attributes, "Random Forest", rf_file_name, font, title_font, target)


	combined_full_dict =  combine_rankings(lasso_dictionary, mrmr_dictionary, random_forest_dictionary, orthogonal_dictionary, path_to_plot_folder)

	graph_combine_plot(combined_full_dict, path_to_plot_folder, figure_size, "cornflowerblue", sensitive_attributes, "Combined_Ranking_"+target+"_Graph", \
						font, title_font, target)

	return "Finished Plotting"



