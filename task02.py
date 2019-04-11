import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def get_data(filename):
    colnames = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholesterol', 'blood_sugar',
                'electrocardiographic_result', 'max_heart_rate', 'exercise_induced', 'old_peak', 'slope',
                'major_vessels', 'thal', 'target']

    return pd.read_csv(filename, sep=',', header=None, names=colnames)


def data_cleansing(dataframe):
    dataframe = dataframe[dataframe.major_vessels != '?']
    dataframe = dataframe[dataframe.thal != '?']

    return dataframe


def feature_importance():
    # create dataframe from csv file and clean the data
    df = data_cleansing(get_data('processed.cleveland.data'))
    df.major_vessels = pd.to_numeric(df.major_vessels)
    df.thal = pd.to_numeric(df.thal)

    # list of attributes apart from the target
    x = df.drop('target', axis=1)

    # target attribute
    y = df.target

    # fit random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x, y)

    # sort features based on their relative importance to the target
    importance_scores = sorted(zip(x.columns, clf.feature_importances_), key=lambda k: k[1], reverse=True)

    return importance_scores


if __name__ == '__main__':
    feature_scores = feature_importance()

    for row in feature_scores:
        print(row)
