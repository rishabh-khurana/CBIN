import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from a import get_data
from task02 import data_cleansing

def load_data(file_path,split_percentage):
    #use get_data method
    df=get_data(file_path)
    df=data_cleansing(df)
    
    df = shuffle(df)
    iris_x = df.drop('target', axis=1).values
    iris_y = df['target'].values

    split_point = int(len(iris_x) * split_percentage)
    iris_X_train = iris_x[:split_point]
    iris_y_train = iris_y[:split_point]
    iris_X_test = iris_x[split_point:]
    iris_y_test = iris_y[split_point:]

    return iris_X_train, iris_y_train, iris_X_test, iris_y_test

    

if __name__ == '__main__':
    file_path='processed.cleveland.data'

    # Split the data into test and train parts
    iris_X_train, iris_y_train, iris_X_test, iris_y_test = load_data(file_path, split_percentage=0.9)

    # train a classifier
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)

    # predict the test set
    predictions = knn.predict(iris_X_test)
    
    print("Actual: ")
    print(iris_y_test)

    print("Predictions: ")
    print(predictions)
