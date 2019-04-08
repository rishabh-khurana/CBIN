import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from task02 import data_cleansing,get_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC,LinearSVC
import numpy as np

def load_data(file_path,split_percentage):
    #use get_data method
    df=get_data(file_path)
    df=data_cleansing(df)
    # the shuffle function alters the data seq
    df = shuffle(df)
    iris_x = df.drop('target', axis=1).values
    iris_y = df['target'].values

    split_point = int(len(iris_x) * split_percentage)
    iris_X_train = iris_x[:split_point]
    iris_y_train = iris_y[:split_point]
    iris_X_test = iris_x[split_point:]
    iris_y_test = iris_y[split_point:]

    return iris_X_train, iris_y_train, iris_X_test, iris_y_test

# 0 incurrs person is healthy
# 1 incurrs person is infected 
def assign_labels(row):
    if row['target']==0:
        return 0
    else:
        return 1

# assign labels to exisitng data frame based on target values
def load_labelled_data(split_percentage):
    df=get_data(file_path)
    df=data_cleansing(df)
    df=shuffle(df)
    # add column label which is derived from target
    df['label']=df.apply(lambda row: assign_labels(row), axis=1)
    # remove label and target columns as they won't be a part of user input
    input_data=df.drop(['label','target'], axis=1).values
    output_data=df['label'].values
    # define split point
    split_point = int(len(input_data) * split_percentage)
    # split data to train and test values
    input_train=input_data[:split_point]
    output_train=output_data[:split_point]
    input_test=input_data[split_point:]
    output_test=output_data[split_point:]

    return input_train,output_train,input_test,output_test

# UserData should be a list of user inputs
# for eg [34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]
def predict_data(UserData):
    input_train, output_train, _, _ = load_labelled_data(split_percentage=0.99)
    GNB=GaussianNB()
    GNB.fit(input_train,output_train)
    return GNB.predict(list(UserData))

# return the accurcy of each classifier
def accuracy_analysis():
    iris_X, iris_y, _, _ = load_labelled_data(split_percentage=1)
    classifiers = [KNeighborsClassifier(),
                   DecisionTreeClassifier(),
                   LinearDiscriminantAnalysis(),
                   LogisticRegression(solver="lbfgs",max_iter=2000),
                   GaussianNB(priors=None, var_smoothing=1e-09),
                   SGDClassifier(max_iter=1000, tol=1e-3)]
    classifier_accuracy_list = []
    for i, classifier in enumerate(classifiers):
        # split the dataset into 5 folds; then test the classifier against each fold one by one
        accuracies = cross_val_score(classifier, iris_X, iris_y, cv=10)
        classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))
    classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
    for item in classifier_accuracy_list:
        print(item[1], ':', item[0])

if __name__ == '__main__':
    file_path='processed.cleveland.data'

    input_train, output_train, input_test, output_test = load_labelled_data(split_percentage=0.995)
    GNB=GaussianNB(priors=None, var_smoothing=1e-09)


    GNB.fit(input_train,output_train)

    print("Expected")
    print(output_test)

    print("Actual")
    print(GNB.predict(input_test))

    accuracy_analysis()