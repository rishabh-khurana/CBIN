import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_data(filename):
    colnames = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholesterol', 'blood_sugar',
                'electrocardiographic_result', 'max_heart_rate', 'exercise_induced', 'old_peak', 'slope',
                'major_vessels', 'thal', 'target']

    return pd.read_csv(filename, sep=',', header=None, names=colnames)


def data_cleansing(dataframe):
    dataframe = dataframe[dataframe.major_vessels != '?']
    dataframe = dataframe[dataframe.thal != '?']

    dataframe.major_vessels = pd.to_numeric(dataframe.major_vessels)
    dataframe.thal = pd.to_numeric(dataframe.thal)

    return dataframe


def feature_importance():
    # create dataframe from csv file and clean the data
    df = data_cleansing(get_data('processed.cleveland.data'))

    # list of attributes (excluding target)
    x = df.drop('target', axis=1)

    # target attribute
    y = df.target

    # # hyperparameter tuning
    # ht = hyperparameter_tuning(x, y)
    # clf = RandomForestClassifier(n_estimators=ht['n_estimators'],
    #                              max_depth=ht['max_depth'],
    #                              min_samples_leaf=ht['min_samples_leaf'],
    #                              min_samples_split=ht['min_samples_split'],
    #                              criterion='gini')

    # fit random forest classifier
    clf = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=4, min_samples_split=5)
    clf.fit(x, y)

    # sort features based on their relative importance to the target
    importance_scores = sorted(zip(x.columns, clf.feature_importances_), key=lambda k: k[1], reverse=True)

    return importance_scores


def hyperparameter_tuning(x, y):
    param_grid = {
        'max_depth': [20, 30, 40, 50],
        'min_samples_leaf': [4],
        'min_samples_split': [5],
        'n_estimators': [200, 400, 600, 800, 1000]
    }

    clf = RandomForestClassifier()

    # 3-fold cross validation for each combination of parameters
    clf_random = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, iid=False)
    clf_random.fit(x, y)

    return clf_random.best_params_


def treemap_json():
    data = feature_importance()
    treemap_data = []

    # labels for treemap
    labels = {
        'age': 'Age',
        'sex': 'Sex',
        'chest_pain': 'Chest Pain Type',
        'blood_pressure': 'Resting Blood Pressure',
        'serum_cholesterol': 'Serum Cholesterol',
        'blood_sugar': 'Fasting Blood Sugar',
        'electrocardiographic_result': 'Resting Electrocardiographic Results',
        'max_heart_rate': 'Maximum Heart Rate',
        'exercise_induced': 'Exercise Induced Angina',
        'old_peak': 'Oldpeak',
        'slope': 'Slope of Peak Exercise ST Segment',
        'major_vessels': 'Major Vessels Coloured by Fluoroscopy',
        'thal': 'Thalassemia'
    }

    score_list = []

    for d in data:
        score_list.append({
            'name': labels[d[0]],
            'value': round(100 * d[1], 4),
            'colorValue': round(100 * d[1], 4)
        })

    # json response for highcharts implementation
    record = {
        'title': {
            'text': 'Attributes Treemap based on Relative Importance to Model (%)'
        },

        'colorAxis': {
            'minColor': '#FFFFFF',
            'maxColor': '#7BCBC3'
        },

        'series': [{
            'type': 'treemap',
            'layoutAlgorithm': 'squarified',
            'data': score_list
        }]
    }

    treemap_data.append(record)

    return treemap_data


# if __name__ == '__main__':
#     feature_scores = treemap_json()
#
#     for row in json.loads(feature_scores[0])['series'][0]['data']:
#         print(f'{row["name"]}: {row["value"]}')
