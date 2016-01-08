import os
import sys
import argparse
from argparse import ArgumentParser
import pandas as pd
import time
from subprocess import call
import time
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

import pickle
import logging

def purge():
	#I should double check using this
	call(["sudo","purge"])
	return None

def build_parser():
	parser = ArgumentParser(
		description="Runs FairML and associated scripts")
	parser.add_argument(
		'--file', type=str, help='input file', dest='input_file', 
		required=True)
	parser.add_argument(
		'--target', type=str,
		help='column name of target variable', dest='target', required=True)
	parser.add_argument(
		'--bootstrap', type=int,
		help='No. of boostrap iterations', dest='no_bootstrap')
	parser.add_argument(
		'--generate_pdf', type=bool, default=False,
		help='generate explanatory pdf', dest='generate_pdf')
	parser.add_argument(
		'--data_bias', type=bool, default=False,
		help='bias vs accuracy plot', dest='data_bias')
	parser.add_argument(
		'--explain', type=bool, default=False,
		help='partial dependence plot for top features', 
		dest='explain_top_features')
	parser.add_argument(
		'--sensitive', nargs='+', type=list,
		help='list of sensitive variables', 
		dest='sensitive_variable_list')

	options = parser.parse_args()
	return options

def transform_column_float(x):
	try:
		return float(x)
	except:
		print "Input file contains characters that can't be converted to float, pls double \
			the input file"
		raise

def create_analysis_folders(options):

	print "setting up variables"
	now = time.time() 
	now_date_format  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
	new_date = "_".join(now_date_format.split())
	name_of_folder = "fairml_analysis_" + new_date

	text_for_read_me = "This folder includes results for the analysis of the file {0}. \
						This analysis was conducted by FairML at {1}.".format(options.input_file, now_date_format)

	#create the names of the folders
	mrmr_output = name_of_folder + "/mrmr_output"
	ranking_results = name_of_folder + "/ranking_results"
	plots = name_of_folder + "/plots"
	log_file = name_of_folder + "/log_file_" + str(now) + ".txt"
	mrmr_input = name_of_folder + "/mrmr_input"

	print "creating folders"
	#create the folders
	call(["mkdir", name_of_folder])
	call(["mkdir", mrmr_output])
	call(["mkdir", mrmr_input])
	call(["mkdir", ranking_results])
	call(["mkdir", plots])
	call(["touch", log_file])

	#initialize log files
	#will deal with log files later
	touch_file = name_of_folder + "/readme.txt"
	call(["touch", touch_file])
	e = "echo {0} >> {1}".format(text_for_read_me, touch_file)
	call(e, shell=True)

	return {"main_folder": name_of_folder, "mrmr_output": mrmr_output, "ranking_results" : ranking_results, 
			"plots": plots, "touch_file":touch_file, "mrmr_input":mrmr_input, "log_file":log_file}

def confirm_input_arguments_and_set_analysis_folders(options):

	print "checking input file"
	#check input file
	try:
		f = open(options.input_file)
	except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise

	print "checking if file can be read by pandas"
	#now read the file and check if target variable is in there. 
	try:
		full_input_data = pd.read_csv(filepath_or_buffer=options.input_file, sep=',')
	except:
		print "Unexpected error while reading the input file", sys.exc_info()[0]
		raise

	#now input file read as csv
	#get list of column names
	column_names = list(full_input_data.columns)
	duplicate_columns = len(column_names) - len(set(column_names))

	print "checking duplicate columns"
	#check for duplicate attributes
	if duplicate_columns > 0:
		raise "Your Input File has duplicate attributes, please remove duplicates"

	print "checking if specified target is in input file"
	#check if target is in list of column names
	column_names_lower = [a.lower() for a in column_names]
	if options.target.lower() not in column_names_lower:
		raise "Your input file does not contain the target variable that you specificied. Please check \
			 that you have the correct spelling "

	target_name_in_csv = column_names[column_names_lower.index(options.target.lower())]
	#check that the sensitive attributes are in the list of variables
	if options.sensitive_variable_list != None:
		sensitive = [a.lower() for a in options.sensitive_variable_list]
		for attribute in sensitive:
			if attribute not in column_names_lower:
				raise "Your input file does not contain the %s that you specificied. Please check \
			 	that you have the correct spelling ", attribute


	print "dropping n/a from dataframe"

	#drop rows with no na
	full_input_data.dropna(axis=0, how='any', inplace=True)


	print "converting all columns to floats"
	#now convert all the columns to float64
	for name in column_names:
		print "we are working with {0} now".format(name)
		if full_input_data[name].dtype != float:
			full_input_data[name] = full_input_data[name].map(transform_column_float)

	print "done converting all columns to floats"

	print "creating analysis folders"
	#set up the relevant analysis folders. 
	folder_paths_and_target = create_analysis_folders(options)

	#now write input file for mrmr analysis
	#ricci_new_df.to_csv(path_or_buf="ricci_data_processed.csv", sep=',', index=False)

	where_to_write_mrmr_input = folder_paths_and_target["mrmr_input"] + "/input_file.csv"
	full_input_data.to_csv(path_or_buf=where_to_write_mrmr_input , sep=',', index=False)

	#and the name of target in the csv into the dictionary
	folder_paths_and_target["target"] = target_name_in_csv
	return full_input_data, folder_paths_and_target


def sample_data_frame_return_x_y(dataframe, contains_y, y_variable_name, num_samples):

	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		X = new_dataframe.drop([y_variable_name],inplace=True, axis=1).values
		return X, y
	else:
		new_dataframe = dataframe.sample(n=num_samples)
		return X, None

def main():
	now = time.time()
	options = build_parser()
	purge()
	input_data_frame, analysis_file_paths = confirm_input_arguments_and_set_analysis_folders(options)
	print "finished writing file to mrmr input"
	csv_input_mrmr = analysis_file_paths["mrmr_input"] + "/input_file.csv"
	print "calling mrmr routing now"
	call_mrmr_routine(csv_input_mrmr, analysis_file_paths["target"], analysis_file_paths["mrmr_output"]+"/")
	purge()
	print "done calling mrmr routing now\n"

	print "aggregating mrmr output"
	aggregate_mrmr_results_and_pickle_dictionary(analysis_file_paths["mrmr_output"], analysis_file_paths["ranking_results"])
	purge()

	print "removing mrmr input folder"

	remove_mrmr_input_folder_to_clean_up_space(analysis_file_paths["mrmr_input"])
	purge()

	print "now we are on random forest"
	best_clf, column_list_for_fit_data = return_best_rf_regressor(input_data_frame, analysis_file_paths["target"], 15, 100, 3)
	purge()

	obtain_feature_importance_from_rf(best_clf, column_list_for_fit_data, analysis_file_paths["ranking_results"])
	purge()

	print "fitting lasso"
	clf, column_list = run_lasso_on_input(input_data_frame, analysis_file_paths["target"])
	purge()

	print "Fitting Lasso took  ------>>> " + str(float(time.time() - now)/60.0) + " minutes!"

	print "Now print feature importance"

	obtain_feature_importance_from_lasso(clf, column_list, analysis_file_paths["ranking_results"])
	purge()
	
	print "Entire analysis took ------>>> " + str(float(time.time() - now)/60.0) + " minutes!"


if __name__ == '__main__':
	main()