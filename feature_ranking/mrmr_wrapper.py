#this file contains functions to call the mrmr routine. 

import os
import sys
import subprocess


def call_mrmr_routine(path_to_csv_file_for_mrmr, target_variable, path_to_write_results):
	#todo: add in a variable indicating path to the mrmr_feature_selection.R script, incase I move it around. 
	subprocess.call(['Rscript', 'mrmr_feature_selection.R', path_to_csv_file_for_mrmr, target_variable, path_to_write_results])
	print "Done with call mrmr routine"
	return "done"

def remove_mrmr_input_folder_to_clean_up_space(file_path_to_mrmr_input_folder):
	print "Removing MRMR Input Folder"
	command = "rm -rf {0}".format(file_path_to_mrmr_input_folder)
	subprocess.call(command, shell=True)
	print "Completely Removed MRMR Input Folder (Hopefully it frees up space!!)"


def recreate_mrmr_input_folder_with_new_dataframe(X):
	#X = pandas dataframe
	call(["mkdir", mrmr_input])

	where_to_write_mrmr_input = "mrmr_input/input_file.csv"

	X.to_csv(path_or_buf=where_to_write_mrmr_input , sep=',', index=False)

	#file path to new mrmr input folder
	return where_to_write_mrmr_input