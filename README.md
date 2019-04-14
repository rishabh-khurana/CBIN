# COMP9321 Assignment 3

Project Description Goes Here

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

Note: Incase above command throws an error 
You can use the following set of commands for running the app:

For Linux and Mac:

```
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run

```

For Windows:

```
set FLASK_APP=flaskr
set FLASK_ENV=development
flask run

```

The app will be served on localhost:5000

## How To Run Task 3 Predict Function

The task3.py file is equipped with predict_data() function

run the predict_data() with predict_data(UserData) argument after running task3.py as a module.

```
>> predict_data([[34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]])
[1]
```
The function will return a value which signifies person has disease or not.

[1]=Person has heart disease
[0]=Person is disease free

