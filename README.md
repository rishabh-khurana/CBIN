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

## To install all packages

```
$ pip3 install -r requirements.txt

```

## How To Run Code
Note:Make sure you are at project directory a.k.a CBIN NOT flaskr

```
> python3 start.py

```

Note: Incase above command throws an error 
You can use the following set of commands for running the app:

For Linux and Mac:

```
export FLASK_APP=flaskr
export FLASK_ENV=development
python3 -m flask run

```

For Windows:

```
set FLASK_APP=flaskr
set FLASK_ENV=development
flask run

```
Note: The app may take a few seconds to run for the first time when you input the aforementioned commands.

The app will be served on localhost:5000

## How To Run Task 3 Predict Function

The task3.py file is equipped with predict_data() function

run the predict_data() with predict_data(UserData) argument after running task3.py as a module.
The UserData arg is a list of list of values input by user. 

```
>> predict_data([[34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]])
[1]
```
The function will return a value which signifies person has disease or not.

[1]=Person has heart disease
[0]=Person is disease free
