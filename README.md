# Question Type Identification
This python code identifies the question type out of the four predefined categories (Who, What, When, Affirmation (yes/no)). If the Label does not does not fall in any of the above then mark that sentence as "Unknown" type.

## Dependencies:
1)PyPhon3.6
2)Pandas 
3)NLTK
4)Sklearn

## Running Steps 
1) Change the train and test file names in main function in "datalabler.py"
2) Run datalabler.py

## Files: 
1) datalabler.py-> code file
2) train.txt-> training data file
3) test.txt-> test file (downloaded from http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label)
4) result.csv-> predictions obtained on above test data
