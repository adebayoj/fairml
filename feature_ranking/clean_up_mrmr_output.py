import os
import sys
import subprocess
import csv
import pickle

#simple float conversion function
def convert_to_float(x):
	try:
		return float(x)
	except:
		return 0.0

test_path = "fairml_analysis_2016-01-04_09:44:13/mrmr_output"

def get_list_of_files(file_path_to_mrmr_output):
	try:
		return os.listdir(file_path_to_mrmr_output)
	except OSError:
		raise "No such file or directory: " + str(file_path_to_mrmr_output)

def write_out_rankings(file_path_to_mrmr_output):

	list_of_files = get_list_of_files(file_path_to_mrmr_output)

	features_index_list = {}
	rankings_dict = {}
	scores = {}
	master_dictionary = {}

	for current_file in list_of_files:
		full_file_path = file_path_to_mrmr_output + "/" + current_file
		if current_file == "feature_names_and_actual_index.csv":
			with open(full_file_path) as infile_index:
				reader = csv.reader(infile_index, delimiter=',')
				next(reader, None)
				for row in reader:
					features_index_list[row[0]] = row[1]

		elif "rank" in current_file:
			#first split by .
			string_current_run = current_file.split(".")[0].split("_")[2] #this gets the number portion from the file name
			list_ranking = []
			with open(full_file_path) as rank_file:
				reader = csv.reader(rank_file, delimiter=',')
				next(reader, None)
				for row in reader:
					list_ranking.append(str(row[1]))
			rankings_dict[string_current_run] = list_ranking

		elif "scores" in current_file:
			string_current_run = current_file.split(".")[0].split("_")[2]
			list_scoring = []
			with open(full_file_path) as score_file:
				reader = csv.reader(score_file, delimiter=',')
				next(reader, None)
				for row in reader:
					value = convert_to_float(row[1])
					if value == float('-inf'):
						value = 0.0
					list_scoring.append(value)
			scores[string_current_run] = list_scoring
		else:
			print "random file found  " + str(current_file)

	#initialize master dictionary
	for key in features_index_list:
		master_dictionary[features_index_list[key]] = 0.0

	print scores

	print "#####################################"

	print rankings_dict

	for number_run in rankings_dict:

		for i in range(len(rankings_dict[number_run])):
			current_rank = rankings_dict[number_run][i]
			current_score = scores[number_run][i]
			variable_name = features_index_list[current_rank]

			#now update master dictionary
			master_dictionary[variable_name] += current_score

	print str(len(rankings_dict)) + " <<---------- length of ranking dictionary"
	#scale each score
	for key in master_dictionary:
		master_dictionary[key] = master_dictionary[key]/len(rankings_dict)
		
	return master_dictionary

def aggregate_mrmr_results_and_pickle_dictionary(file_path_to_mrmr_output, file_path_to_rankings):

	output_dict = write_out_rankings(file_path_to_mrmr_output)

	pickle_path = file_path_to_rankings + "/mrmr_feature_ranking.pickle"

	with open(pickle_path, 'wb', ) as handle:
  		pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  	return "Pickled mrmr feature rankings"
