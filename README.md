# COMP9321 Assignment 3

## Project Description
This assignment produces a web application which contains different analysis/task on the ”processed.cleveland.data” data set.

The main functions of the web application are the following:
1) Visualise the statistics for basic statistics of a dataset by groups of
age and sex. The statistics are visualised using group stacked bar chart and histogram analysis is used to properly bin the data

2) Visualise the importance of each statistics in relation to a person having heart disease. The importance is visualised using a treemap and random forest algorithm is used to rank importance of each attribute

3) Predict if a person has heart disease or not based on users input in a web form. The prediction is mainly based on naive baiyes classifier with some optimisation to tune up the accuracy and to reduce overfitting. The accurracy is also projected in the line graph below the prediction results.



## Getting Started

This is set up with only Flask 1.0.2 and SQLite

Database is linked and set up


Folder structure:

flaskr/static for any css, js, img files (can split further into subfolder)

flaskr/templates for html files (those used for corresponding .py is in their subfolder)

## How To Run Code
Make sure you are at project directory a.k.a CBIN NOT flaskr

```
$ python start.py
```
The app will be served on localhost:5000

## What To Play With ATM
Go to localhost:5000/chart/index 

Enter 1 for Template Id field

Any text for Random Text field

=> Direct to Template 1 page
