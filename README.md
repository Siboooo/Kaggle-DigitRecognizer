# Kaggle-Competition-DigitRecognizer

## Overview
This is my first kaggle competition project, also my first `Python` program. In this project, the goal is to correctly identify the handwritten single digit by training a dataset of tens of thousands of handwritten images. More information about the competition can be found [here](https://www.kaggle.com/c/digit-recognizer#description).

## Details
This program utilizes a Multi-Layer Perceptron (Neural Network with one hidden layer) to achieve the goal.

Training dataset and test dataset can be downloaded [here](https://www.kaggle.com/c/digit-recognizer/data).

A small training set and a small test set are provided in this project for testing. 

Data augmentation --> Rotated images are used to expand the training dataset.

## Performance
Four versions (with different learning rates, hidden unit numbers and epoch numbers) were submitted (Score up to 97.5%).

![submission](https://raw.githubusercontent.com/Siboooo/imgForMD/master/DigitRecognizer/DR-MLP-sub.png) 

## Dependencies
* [NumPy](http://www.numpy.org)
* [SciPy](https://www.scipy.org)
* [Pandas](http://pandas.pydata.org)
