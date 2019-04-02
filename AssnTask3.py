import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from task02 import data_cleansing,get_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# UserData should be a list of user inputs
# for eg [34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]
def predict_data(UserData):
    iris_X_train, iris_y_train, _, _ = load_data(file_path, split_percentage=0.99)
    lda=LinearDiscriminantAnalysis()
    lda.fit(iris_X_train,iris_y_train)
    return lda.predict(list(UserData))

    

if __name__ == '__main__':
    file_path='processed.cleveland.data'

    # Split the data into test and train parts
    iris_X_train, iris_y_train, iris_X_test, iris_y_test = load_data(file_path, split_percentage=0.99)

    # train a classifier
    #knn = KNeighborsClassifier()
    #knn.fit(iris_X_train, iris_y_train)

    # predict the test set
    #predictions = knn.predict(iris_X_test)

    lda=LinearDiscriminantAnalysis()
    lda.fit(iris_X_train,iris_y_train)
    #predictions = lda.predict([[34.0,1.0,1.0,118.0,182.0,0.0,2.0,174.0,0.0,0.0,1.0,0.0,3.0]])
    predictions=lda.predict(iris_X_test)
    print("Actual: ")
    print(iris_y_test)

    print("Predictions: ")
    print(predictions)

    print("For data")
    print(iris_X_test)