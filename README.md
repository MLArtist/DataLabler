# Question Type Identification
This python code identifies the question type out of the four predefined categories (Who, What, When, Affirmation (yes/no)). If the Label does not does not fall in any of the above then mark that sentence as "Unknown" type.

## Dependencies:
- PyPhon3.6
- Pandas 
- NLTK
- Sklearn

## Running Steps 
1) Change the train and test file names in main function in "datalabler.py"
2) Run datalabler.py

## Files: 
- datalabler.py-> code file
- train.txt-> training data file
- test.txt-> test file (downloaded from http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label)
- result.csv-> predictions obtained on above test data
