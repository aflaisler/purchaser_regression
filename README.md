# purchaser purchasers prediction

# General description

- Two python scripts (pre_processing.py and purchaser_prediction.py)
- One folder (./Data)

- The scripts are used to preprocess the data and then make a prediction about the users
- The folder is used to store the data

# How to use

----------
Preparation
----------

- Put the tsv files in the ./Data folder

----------
1st step
----------
- In the command line, type :

	python pre_processing.py

- This will preprocess the data

----------
2nd step
----------

- In the command line, type :

	python purchaser_prediction.py

- This will create two csv files in ./Data
- One file gives the result for all the users in the test.tsv file
- One file gives only the potential purchasers


# Details of the preprocessing script

- This script splits the Events file into several smaller files
- Each file deals with a specific event category (web visits, web_pv ...)
- This is to allow faster execution/loading and better organization of the project
- The files are stored in ./Data/Events/

# Details of the purchaser_prediction script

- This script prepare the data for a logistic regression
- It then run the regression for the training set and applies it to the test data.

# Next steps

There are many different steps that could be tried in order to improve the model:

- Removing features (Using Forward Stepwise Regression)
- Apply Bootstrap Aggregation (“Bagging”)
- Test Non-linear models
