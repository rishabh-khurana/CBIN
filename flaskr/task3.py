import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from flaskr.task2 import data_cleansing, get_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC,LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
import json

def load_data(file_path,split_percentage):
    #use get_data method
    df=get_data(file_path)
    df=data_cleansing(df)
    # the shuffle function alters the data seq
    df = shuffle(df)
    input_val = df.drop('target', axis=1).values
    output_val = df['target'].values

    split_point = int(len(input_val) * split_percentage)
    input_train = input_val[:split_point]
    output_train = output_val[:split_point]
    input_test = input_val[split_point:]
    output_test = output_val[split_point:]

    return input_train, output_train, input_test, output_test

# 0 incurrs person is healthy
# 1 incurrs person is infected 
def assign_labels(row):
    if row['target']==0:
        return 0
    else:
        return 1

# assign labels to exisitng data frame based on target values
def load_labelled_data(split_percentage):
    file_path='processed.cleveland.data'
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
# for eg [[34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]]
def predict_data(UserData):
    input_train, output_train, _, _ = load_labelled_data(split_percentage=0.99)
    GNB=GaussianNB()
    GNB.fit(input_train,output_train)
    #print(GNB.predict(UserData).tolist())
    # return values are either [0] or [1] depending if person is healthy or infected
    return (GNB.predict(UserData))

# return the accurcy of each classifier
def accuracy_analysis():
    iris_X, iris_y, _, _ = load_labelled_data(split_percentage=1)
    classifiers = [KNeighborsClassifier(),
                   DecisionTreeClassifier(),
                   LinearDiscriminantAnalysis(),
                   LogisticRegression(solver="lbfgs",max_iter=2000),
                   GaussianNB(priors=None, var_smoothing=1e-09),
                   SGDClassifier(max_iter=1000, tol=1e-3),
                   MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)]
    classifier_accuracy_list = []
    for i, classifier in enumerate(classifiers):
        # split the dataset into 5 folds; then test the classifier against each fold one by one
        accuracies = cross_val_score(classifier, iris_X, iris_y, cv=10)
        classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))
    classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
    for item in classifier_accuracy_list:
        print(item[1], ':', item[0])
    return classifier_accuracy_list

def plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    # assign labels to coordinates
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def build_json(estimator=GaussianNB(),n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5)):
    X, y, _, _ = load_labelled_data(split_percentage=0.90)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    result=[]
    obj={}
    
    title={}
    title['text']=type(estimator).__name__
    obj['title']=title

    xAxis={}
    title['text']='Accuracy'
    xAxis['title']=title
    obj['xAxis']=xAxis

    yAxis={}
    title['text']='Iterations'
    yAxis['title']=title
    obj['yAxis']=yAxis

    #series property
    series=[]
    #line1 data
    line1={}
    line1['name']='Training score'
    data=[]
    for i,j in zip(train_sizes,train_scores_mean):
        data_sub=[]
        data_sub.append(float(i))
        data_sub.append(float(j))
        data.append(data_sub)
    line1['data']=data
    line1['zIndex']=1
    marker={}
    marker['fillColor']='white'
    marker['lineWidth']=2
    marker['lineColor']='red'
    line1['marker']=marker
    line1['lineColor']='red'
    series.append(line1)
    range1={}
    range1['name']='Range1'
    data=[]
    for i,j,k in zip(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std):
        data_sub=[]
        data_sub.append(float(i))
        data_sub.append(float(j))
        data_sub.append(float(k))
        data.append(data_sub)
    range1['data']=data
    range1['type']='arearange'
    range1['lineWidth']=0
    range1['linkedTo']=':previous'
    range1['color']='rgb(196, 10, 0)'
    range1['fillOpacity']=0.2
    range1['zIndex']=0
    marker={}
    marker['enabled']="false"
    range1['marker']=marker
    series.append(range1)
    #line2 data
    line2={}
    line2['name']='Test Score'
    data=[]
    for i,j in zip(train_sizes,test_scores_mean):
        data_sub=[]
        data_sub.append(float(i))
        data_sub.append(float(j))
        data.append(data_sub)
    line2['data']=data
    line2['zIndex']=2
    marker={}
    marker['fillColor']='white'
    marker['lineWidth']=2
    marker['lineColor']='green'
    line2['marker']=marker
    line2['lineColor']='green'
    series.append(line2)
    range2={}
    range2['name']='Range2'
    data=[]
    for i,j,k in zip(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std):
        data_sub=[]
        data_sub.append(float(i))
        data_sub.append(float(j))
        data_sub.append(float(k))
        data.append(data_sub)
    range2['data']=data
    range2['type']='arearange'
    range2['lineWidth']=0
    range2['linkedTo']=':previous'
    range2['color']='rgb(19, 196, 0)'
    range2['fillOpacity']=0.2
    range2['zIndex']=0
    marker={}
    marker['enabled']="false"
    range2['marker']=marker
    series.append(range2)

    obj['series']=series
    # obj=json.dumps(obj)
    result.append(obj)
    return result

if __name__ == '__main__':
    file_path='processed.cleveland.data'

    # input_train, output_train, input_test, output_test = load_labelled_data(split_percentage=1)
    estimator = GaussianNB()
    # title = "Learning Curves (Naive Bayes)"
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # plot_learning_curve(DecisionTreeClassifier(), title, input_train, output_train, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    # plt.show()
    '''estimators=[KNeighborsClassifier(),
                   DecisionTreeClassifier(),
                   LinearDiscriminantAnalysis(),
                   LogisticRegression(solver="lbfgs",max_iter=2000),
                   GaussianNB(priors=None, var_smoothing=1e-09),
                   SGDClassifier(max_iter=1000, tol=1e-3),
                   ]'''
    '''To use the build_json function use any estimator mentioned above'''
    #For eg. build_json(GaussianNB())
    json_data=build_json(estimator)
    print(json_data)