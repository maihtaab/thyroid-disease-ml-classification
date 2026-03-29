import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path, header=None)

def preprocess(df):
    df = df.replace("?", np.nan)

    # real target is the LAST column, before the |
    raw_target = df.iloc[:, -1].astype(str).str.split("|").str[0].str.replace(".", "", regex=False).str.strip()

    # binary target: hyperthyroid-related vs negative
    positive_labels = {"hyperthyroid", "T3 toxic", "goitre"}
    y = raw_target.apply(lambda x: 1 if x in positive_labels else 0)

    # features = everything except referral source? no, keep referral source, drop only final label|id column
    X = df.iloc[:, :-1].copy()

    # convert what can be numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="ignore")

    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(exclude=["number"]).columns

    # drop fully empty numeric cols
    drop_cols = [col for col in numeric_cols if X[col].isna().all()]
    X = X.drop(columns=drop_cols)

    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(exclude=["number"]).columns

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        mode_val = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode_val[0] if not mode_val.empty else "missing")

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.fillna(0)
    X.columns = X.columns.astype(str)

    return X, y