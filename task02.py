import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier


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


def treemap_json():
    data = feature_importance()
    treemap_data = []

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
    cv = len(data)
    for d in data:
        score_list.append({
            'name': labels[d[0]],
            'value': f'{int(10000 * d[1]) / 100}%',
            'colorValue': cv
        })

        cv -= 1

    # json for highcharts implementation
    record = {
        'title': {
            'text': 'Attributes Treemap based on Importance to Model'
        },

        'colorAxis': {
            'minColor': '#FFFFFF',
            'maxColor': '#5522FF'
        },

        'series': [{
            'type': 'treemap',
            'layoutAlgorithm': 'squarified',
            'data': score_list
        }]
    }

    json_record = json.dumps(record)
    treemap_data.append(json_record)

    return treemap_data


# if __name__ == '__main__':
#     feature_scores = treemap_json()
#
#     print(feature_scores)
