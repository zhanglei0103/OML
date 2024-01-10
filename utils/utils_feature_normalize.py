import sys
import os

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn import preprocessing


def feature_preprocess(X, preprocess_method):

    if X.ndim == 1:
        cols = list(X.name)
        rows = list(X.index)
        X = np.array(X).reshape(-1, 1)
    else:
        cols = list(X.columns)
        rows = list(X.index)

    if preprocess_method == "StandardScaler":
         scaler = preprocessing.StandardScaler()
    elif preprocess_method == "MinMaxScaler":
        scaler = preprocessing.MinMaxScaler()
    elif preprocess_method == "MaxAbsScaler":
        scaler = preprocessing.MaxAbsScaler()
    elif preprocess_method == "RobustScaler":
        scaler = preprocessing.RobustScaler()
    elif preprocess_method == "QuantileTransformer":
         scaler = preprocessing.QuantileTransformer(output_distribution='normal')
    elif preprocess_method == "PowerTransformer":
         scaler = preprocessing.PowerTransformer(method='box-cox')
    else:
        raise NotImplementedError(f'Preprocessing method ({preprocess_method}): not implemented')

    X_scale = scaler.fit_transform(X)
    X_scale = pd.DataFrame(X_scale, columns=cols, index=rows)
    return X_scale


def feature_trans(df, feature, feature_method):
    if (feature_method is None) | (feature_method == 'None'):
        df[feature] = df[feature]
    elif feature_method == 'LOG':
        df[feature] = np.log(df[feature])
    elif feature_method == 'LOG2':
        df[feature] = np.log2(df[feature])
    elif feature_method == 'LOG10':
        df[feature] = np.log10(df[feature])
    elif feature_method == 'SQRT':
        df[feature] = np.sqrt(df[feature])
    elif feature_method == 'RECIPROCAL':
        df[feature] = np.reciprocal(df[feature])
    elif feature_method == 'SQUARE':
        df[feature] = np.square(df[feature])
    elif feature_method == 'LOGIT':
        # feature must be defined as proportion
        df[feature] = logit(df[feature].to_numpy())
    elif feature_method == "Z-Score":
        df[feature] = feature_preprocess(df[feature], 'StandardScaler')
    else:
        raise ValueError(f"transformation method ({feature_method}) is not supported\n")
    return df

