#this file contains functions to call the mrmr routine. 

import os
import sys
import subprocess


def call_mrmr_routine(path_to_csv_file_for_mrmr, target_variable, path_to_write_results):
	subprocess.call(['Rscript', 'mrmr_feature_selection.R', path_to_csv_file_for_mrmr, target_variable, path_to_write_results])
	print "Done with call mrmr routine"
	return "done"
