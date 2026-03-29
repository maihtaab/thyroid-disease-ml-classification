import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path, header=None)

    # remove last column after |
    df = df[0].str.split("|", expand=True)[0]
    df = df.str.split(",", expand=True)

    return df


def preprocess(df):

    # replace missing values
    df.replace("?", np.nan, inplace=True)

    # target column
    y = df.iloc[:, -1]

    # remove the period
    y = y.str.replace(".", "", regex=False)

    # convert labels
    y = y.apply(lambda x: 1 if x == "hyperthyroid" else 0)

    # features
    X = df.iloc[:, :-1]

    # encode categorical variables
    X = pd.get_dummies(X)

    # fill missing values
    X = X.fillna(X.mean(numeric_only=True))

    return X, y
