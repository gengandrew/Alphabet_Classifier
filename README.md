# Alphabet Classifier

## About

A simple alphabet classifier trainied on the emnist training set utilizing a naive multivariate normal classifier setup.

## Requirements

The program is intented to run on python 3.7 with the following necessary libraries:
- scipy, numpy, sympy
- matplotlib
- emnist

## Setup

- By default we are taking only 5000 samples from the emnist training set.
    - Alterations can be made through the "trainingSize" variable.
- The algorithm performace varies probabilistically, thus we run 500 epochs taking the best performing covariance.
    - Alterations can be made through the "epochSize" variable.
- Resulting means and covariances will be written out to text files which can be parsed using parse.py.

## Getting started

Run the following to start training the classifier:
````
python3 aclassify.py
````
After altering the naming conventions of the best performaning classifier to "means.txt" and "covars.txt" parse with:
````
python3 parse.py
````
Resulting classifier will be placed into a "model.txt" files with the following structure:
````
[{ 'label': 'a', 'covar': [[]], 'mean': [] } ...]
````
