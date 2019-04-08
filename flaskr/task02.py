import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def get_data(file):
    colnames = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholesterol', 'blood_sugar',
                'electrocardiographic_result', 'max_heart_rate', 'exercise_induced', 'old_peak', 'slope',
                'major_vessels', 'thal', 'target']

    return pd.read_csv(file, sep=',', header=None, names=colnames)


def data_cleansing(df):
    df = df[df.major_vessels != '?']
    df = df[df.thal != '?']

    return df


if __name__ == '__main__':
    df = get_data('processed.cleveland.data')
    df = data_cleansing(df)

    for row in df.itertuples():
        print(row)

    X = df.drop('target', axis=1)

    model = LogisticRegression()
    rfe = RFE(model, 1)

    rfe = rfe.fit(X, df.target)
    print(rfe.support_)
    print(rfe.ranking_)
