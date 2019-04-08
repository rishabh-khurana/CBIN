import pandas as pd
# from sklearn.feature_selection import RFE, SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import LinearSVC


def get_data(filename):
    colnames = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholesterol', 'blood_sugar',
                'electrocardiographic_result', 'max_heart_rate', 'exercise_induced', 'old_peak', 'slope',
                'major_vessels', 'thal', 'target']

    return pd.read_csv(filename, sep=',', header=None, names=colnames)


def data_cleansing(dataframe):
    dataframe = dataframe[dataframe.major_vessels != '?']

    return dataframe[dataframe.thal != '?']


if __name__ == '__main__':
    df = data_cleansing(get_data('processed.cleveland.data'))
    df.major_vessels = pd.to_numeric(df.major_vessels)
    df.thal = pd.to_numeric(df.thal)

    for row in df.itertuples():
        print(row)

    X = df.drop('target', axis=1)
    y = df.target

    # RANDOM FOREST REGRESSOR

    rf = RandomForestRegressor(n_estimators=1000)
    rf.fit(X, y)

    for feature in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True):
        print(feature)

    # # RANDOM FOREST CLASSIFIER
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    # clf.fit(X_train, y_train)
    #
    # for feature in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True):
    #     print(feature)

    # # PRINCIPAL COMPONENT ANALYSIS
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)
    #
    # pca = PCA()
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)
    #
    # explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)

    # # RECURSIVE FEATURE ELIMINATION
    #
    # model = LogisticRegression(solver='liblinear', multi_class='auto')
    # rfe = RFE(model, 1)
    #
    # rfe = rfe.fit(X, df.target)
    # print(rfe.support_)
    # print(rfe.ranking_)
