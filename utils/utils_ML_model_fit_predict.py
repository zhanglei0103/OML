import sys
import os

import pandas as pd
import shap
import streamlit as st
import scipy.stats as stat
import numpy as np
import matplotlib
matplotlib.use('agg')
import copy
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import (roc_curve, auc, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score,
                            confusion_matrix, mean_squared_error, r2_score)
from utils.utils_roc_ci_boostrap import roc_ci_bootstrap
from utils.utils_binarymetric_ci_boostrap import binarymetric_ci_boostrap


def checktype_X_y(X, y, isreshape=False):
    """
    X:training vector(feature vector), where n_samples is the number of samples and n_features is the number of features
    y:target vector relative to X.
    """
    if type(X).__module__ == np.__name__:
        X = X
    else:
        X = X.to_numpy()

    if isreshape:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    if type(y).__module__ == np.__name__:
        y = y
    else:
        y = np.array(y)
    return X, y


class RegressionIntermediateVariable:
    inter_variable = {'y_reals': [],
                      'y_preds': [],
                      'y_reals_cv': [],
                      'y_preds_cv': [],
                      'rmses': [],
                      'rsquareds': [],
                      'rsquareds_mean_std': None,
                      'rmses_mean_std': None,
                      "shap_values_per_cv": [],
                      "shap_values_mean_cv": None,
                      }


def regression_model_direct_predict(inter_variable, y_real, y_pred, rmse, rsquared):
    inter_variable['y_reals'].extend(y_real)
    inter_variable['y_preds'].extend(y_pred)
    inter_variable['y_reals_cv'].append(y_real)
    inter_variable['y_preds_cv'].append(y_pred)
    inter_variable['rmses'].append(rmse)
    inter_variable['rsquareds'].append(rsquared)
    return inter_variable


def regression_ML_model_predict_v2(reg_fit, X, y):
    X, y = checktype_X_y(X, y)

    y_pred = reg_fit.predict(X)
    rmse = mean_squared_error(y_true=y, y_pred=y_pred, squared=False)
    rsqared = r2_score(y_true=y, y_pred=y_pred)
    return y_pred, rmse, rsqared


def mean_std_metric(metric):
    # metric must be list
    if len(metric) > 1:
        mean_metric = np.round(np.mean(metric), 3)
        std_metric = np.round(np.std(metric), 3)
        mean_std = f'{mean_metric}({std_metric})'
    else:
        mean_metric = f'{metric[0]}'
        mean_std = f'{metric[0]}'
    return mean_metric, mean_std



