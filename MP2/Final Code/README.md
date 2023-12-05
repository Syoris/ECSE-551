# ECSE-551 Mini Project 2
Authors:
* Ashley Meagher (260822930)
* Charles Sirois (261158513)  

## Code Execution
Steps to execute the results in Google Colab:

1. Unzip and download the project's folder in your Google Drive
2. In *MP2.ipynb*, at the top of the file: 
   1. Set `in_colab` variable to `True`
   2. Set `folder_path` to the path in your Google Drive where you placed the folder. It should be *`drive/MyDrive/{path_to_project}`*
3. Run the notebook

## Files
* MP2.ipynb: Main file. To produce all the results.
* NaiveBayes.py: Implementation of the Bernoulli Naive Bayes classifier.
* cross_val_score: Function that computes the CV score for all combinations of specified hyperparameters.
* data_processing.py: Implements two classes:
  * `Data`: To load the csv files and identify the language.
  * `Format_data`: To process the dataset. 