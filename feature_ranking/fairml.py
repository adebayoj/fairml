import os
import sys
import argparse
from argparse import ArgumentParser
import pandas as pd
import time

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

def confirm_input_arguments(options):
	#check input file
	try:
		f = open(input_file)
	except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise


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


	#check for duplicate attributes
	if duplicate_columns > 0:
		raise "Your Input File has duplicate attributes, please remove duplicates"

	#check if target is in list of column names
	column_names_lower = [a.lower() for a in column_names]
	if options.target.lower() not in column_names_lower:
		raise "Your input file does not contain the target variable that you specificied. Please check \
			 that you have the correct spelling "

	#check that the sensitive attributes are in the list of variables
	sensitive = [a.lower() for a in options.sensitive]
	for attribute in sensitive:
		if attribute not in column_names_lower:
			raise "Your input file does not contain the %s that you specificied. Please check \
			 that you have the correct spelling ", attribute



	#drop rows with no na
	full_input_data.dropna(axis=0, how='any', inplace=True)


	#now convert all the columns to float64
	for name in column_names:
		if full_input_data[name].dtype != float:
			full_input_data = full_input_data[name].map(transform_column_float)


	#check input variable name
	return options


def sample_data_frame_return_x_y(dataframe, contains_y, y_variable_name, num_samples):

	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		X = new_dataframe.drop([y_variable_name],inplace=True, axis=1).values
		return X, y
	else:
		new_dataframe = dataframe.sample(n=num_samples)
		return X, -

def main():
	options = build_parser()
	print options.input_file

if __name__ == '__main__':
	main()