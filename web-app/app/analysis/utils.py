
import pandas as pd 
import os


from sklearn.preprocessing import StandardScaler

import pickle


def scale_input_data(X):
	 scaler = StandardScaler()
	 scaler.fit(X)
	 X = scaler.transform(X.copy())
	 return X, scaler


def transform_column_float(x):
	try:
		return float(x)
	except:
		print "Input file contains characters that can't be converted to float, pls double \
			the input file"
		raise


def sample_data_frame_return_x_y_column_name(dataframe, contains_y, y_variable_name, num_samples):

	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		new_dataframe.drop([y_variable_name],inplace=True, axis=1)
		column_list = list(new_dataframe.columns)
		return new_dataframe.values, y, column_list
	else:
		raise "Input file to sample should always contain the y variable. "


def sample_data_frame_return_x_y_rf_file(dataframe, contains_y, y_variable_name, num_samples):
	if contains_y:
		new_dataframe = dataframe.sample(n=num_samples)
		y = new_dataframe[y_variable_name].values
		new_dataframe.drop([y_variable_name],inplace=True, axis=1)
		column_list = list(new_dataframe.columns)
		return new_dataframe.values, y, column_list
	else:
		raise "Input file to sample should always contain the y variable. "


def pickle_this_variable_with_this_name_to_this_folder(variable, variable_name_or_path):

	with open(variable_name_or_path, 'wb') as handle:
		pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return "Pickled to the file  to " + variable_name_or_path

def get_list_of_files(file_path):
	try:
		return os.listdir(file_path)
	except OSError:
		raise "No such file or directory: " + str(file_path)


