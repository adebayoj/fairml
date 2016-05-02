" 
Overview of this script

Performs Feature Ranking using the MRMRe Library from CRAN. 

Run the script like this from the commmand line (os x): Rscript mrmr_feature_selection.R arg1 arg2 arg3
example: Rscript mrmr_feature_selection.R code/data/processed_data_sets/turkey_credit_individual_data_with_pd_limit.csv credit_limit testing_results/mrmr/

arg1: file path to the input csv that contains the data
arg2: column name of the target in the input csv
arg3: output file path or folder where the results of the analysis is dumped

OUTPUT
This script outputs a number of files (2*bootstrap_runs + 1) dependending on the number of bootstrap runs
1 - feature_scores_*  = mrmr scores for the features based on the rankings in feature rank
2 - feature_rank_* = mrmr feature rank based on the scores. 

Note: You can run clean_up_mrmr_output.py on the output results from this script to obtain
a condensed dictionary with all the ranking for each feature. 
"

ptm <- proc.time()

#read in mrmr library
library(mRMRe)
set.thread.count(2) #for faster runs

args<-commandArgs(TRUE)

#read command line arguments
data_file_name <- args[1]
target_variable_column_name <- args[2]
path_to_write_results <- args[3]



#test prints
print("printing current directory")
print(getwd())

print("print file name")
print(data_file_name)

#set number of bootstrap samples
no_iter_bootstrap = 10 #MRMRe already has this working internally, but it returns N/A for some reason when I do that, so I am basically doing this myself
bootstrap_fraction = 0.8 #percent of whole data to use when running bootstrap

#read in the data
#todo: wrap this in a try catch statement to figure out if input file is read in 
input_data <- read.csv(data_file_name, header=TRUE)
input_data <- data.frame(input_data)
print("read in input data successfully!")

print("amount of time taken to read in the file")
proc.time() - ptm



#check dimension of the data
print(dim(input_data))

#get data dimensions
no_of_rows = dim(input_data)[1]
no_of_cols = dim(input_data)[2]

random_no_boostrap_samples = round(bootstrap_fraction*no_of_rows) #no of random samples to be drawn from entire file
no_of_features = no_of_cols - 1 #this is because the target is included in the csv file

#make sure all columns of dataframe are numeric
#First get dataframe of all columns and class
#data_frame_columns_and_class <- data.frame(Reduce(rbind, sapply(data.frame(data), class))) #not needed anymore
#batch convert all columns to numeric using this
print("converting all columns to numeric!")
input_data <- sapply(input_data, as.numeric)
input_data <- data.frame(input_data)

print("total amount of time taken till making all columns numeric")
proc.time() - ptm

print("Starting bootstrap sampling for mrmr feature selection")

#for no_iter_bootstrap_times, do mrmr iteration
for(i in 1:no_iter_bootstrap )
{
  print(i)
  
  #get the smaller sample from the larger dataframe.
  print("creating the mrmr data object")
  input_data_sample <- input_data[sample(nrow(input_data), random_no_boostrap_samples), ]
  input_data_sample <- data.frame(input_data_sample)
  
  #get column index of the target variable
  index_of_target_variable <- which(names(input_data_sample)==target_variable_column_name)
  print("got index of target variable !")
  
  #create mrmr data object
  print("creating the mrmr data object")
  feature_data <- mRMR.data(data = input_data_sample)
  
  #create filter object
  print("creating the mrmr filter object")
  filter <- mRMR.classic("mRMRe.Filter", data=feature_data, target_indices = index_of_target_variable, feature_count=no_of_features)
  
  print("calculating scores and solutions")
  feature_scores <- scores(filter)
  feature_rank <- solutions(filter)
  
  print("converting scores and solutions to dataframe")
  data_frame_test_scores <- data.frame(Reduce(rbind, feature_scores))
  data_frame_test_rank <- data.frame(Reduce(rbind, feature_rank))
  
  print("creating file names for results")
  file_path_for_scores <- paste(path_to_write_results, "feature_scores_", sprintf("%04d",i), ".csv", sep="")
  file_path_for_rank <- paste(path_to_write_results, "feature_rank_", sprintf("%04d",i), ".csv", sep="")
  
  print("writing the scores and solutions to a csv file")
  write.table(data_frame_test_scores, file = file_path_for_scores, sep = ",", col.names = NA, qmethod = "double")
  write.table(data_frame_test_rank, file = file_path_for_rank, sep = ",", col.names = NA, qmethod = "double")  
  
  if(i==1)
  {
    #write out the feature names once
    file_path_feature_names <- paste(path_to_write_results, "feature_names_and_actual_index.csv", sep="")
    write.table(data.frame(featureNames(feature_data)), file = file_path_feature_names, sep=",", col.names=NA)
  }
}

print("Time for Feature Ranking below")
proc.time() - ptm

print("done!")


