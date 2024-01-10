import sys
import os

import streamlit as st
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import (SVR, SVC)
from sklearn.neighbors import (KNeighborsRegressor, KNeighborsClassifier)
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

n_jobs = 1

def linear_reg_set_params(ml_method):
    st.info(
        """Wow, no need to set parameters for linear regression. You can click **submit** button below to run!!!""")
    set_model_parameters = dict()
    return set_model_parameters


def lasso_reg_set_params(ml_method):
    col0, col1, col2 = st.columns(3)
    alpha = col0.slider(label="1. regularization strength", min_value=0.0, max_value=10000.0, value=1.0,
                        help="Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).")
    max_iter = col1.slider('2. the maximum number of iterations', 100, 10000, 1000, 100)
    tol = col2.selectbox('3. the tolerance for the optimization',
                         [1e-4, 2, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-6],
                         help="The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol, see Notes below.")
    # selection = st.selectbox('coefficient update', ['cyclic', 'random'])
    set_model_parameters = {
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol
    }
    return set_model_parameters


def ridge_reg_set_params(ml_method):
    col0, col1, col2 = st.columns(3)
    alpha = col0.slider('1. regularization strength', 0.0, 10000.0, 1.0,
                        help="Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).")
    max_iter = col1.slider('2. the maximum number of iterations', 100, 10000, 1000, 100)
    tol = col2.selectbox('3. the tolerance for the optimization',
                         [1e-3, 2, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6],
                         help="The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol, see Notes below.")
    set_model_parameters = {
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol
    }
    return set_model_parameters


def elasticnet_reg_set_params(ml_method):
    col0, col1, col2 = st.columns(3)
    alpha = col0.slider('1. regularization strength', 0.0, 10000.0, 1.0,
                        help="Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).")
    max_iter = col1.slider('2. the maximum number of iterations', 100, 10000, 1000, 100)
    tol = col2.selectbox('3. the tolerance for the optimization',
                         [1e-3, 2, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6],
                         help="The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol, see Notes below.")
    col3, _, _ = st.columns(3)
    l1_ratio = col3.slider('4. l1 ratio', 0.0, 1.0, 0.5, help="""
                In [scikit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html?highlight=elasticnet#sklearn.linear_model.ElasticNet), l1_ratio is the ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
                """)
    set_model_parameters = {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': max_iter,
        'tol': tol
    }
    return set_model_parameters


def bayesian_ridge_set_params(ml_method):
    cols = st.columns(3)
    n_iter = cols[0].slider('1. Maximum number of iterations', 1, 5000, 300,
                            help="Should be greater than or equal to 1. The actual number of iterations to reach the stopping criterion.")
    tol = cols[1].selectbox('2. the tolerance for the optimization', [1e-3, 2, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6],
                            help="Stop the algorithm if convergence existence.")
    alpha_1 = cols[2].selectbox('3. alpha_1', [1e-6, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-8],
                                help="Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter.")
    cols1 = st.columns(3)
    alpha_2 = cols1[0].selectbox('4. alpha_2', [1e-6, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-8],
                                 help="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter.")
    lambda_1 = cols1[1].selectbox('5. lambda_1', [1e-6, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-8],
                                  help="Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter.")
    lambda_2 = cols1[2].selectbox('6. lambda_2', [1e-6, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-8],
                                  help="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter.")
    set_model_parameters = {
        'n_iter': n_iter,
        'tol': tol,
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
    }
    return set_model_parameters


def svr_set_params(ml_method):
    cols = st.columns(3)
    C = cols[0].slider('1. Regularization parameter', 1e-5, 10000.0, 1.0,
                       help="The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
    kernel = cols[1].selectbox('2. Kernel function',
                               ['linear', 'poly', 'rbf', 'sigmoid'],
                               help="Specifies the kernel type to be used in the algorithm.")
    degree = cols[2].number_input('3. degree', 1, 10, 3,
                                  help="Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.")
    cols1 = st.columns(3)
    gamma = cols1[0].selectbox('4. gamma', ['scale', 'auto', 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                               help="Kernel coefficient for rbf, poly and sigmoid")
    tol = cols1[1].selectbox('5. Tolerance for stopping criterion', [1e-3, 1e-2, 1e-4, 1e-5, 1e-6])
    epsilon = cols1[2].slider('6. Epsilon in the epsilon-SVR model', 0.001, 100.0, 0.1, help="It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative.")
    set_model_parameters = {
        'C': C,
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'tol': tol,
        'epsilon': epsilon,
    }
    return set_model_parameters


def knn_regression_set_params(ml_method):
    cols = st.columns(3)
    n_neighbors = cols[0].slider('1. Number of neighbors', 4, 30, 5,
                            help="")
    algorithm = cols[1].selectbox('2. Algorithm used to compute the nearest neighbors', ["auto", "ball_tree", "kd_tree", "brute"],
                            help="‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method. Note: fitting on sparse input will override the setting of this parameter, using brute force.")
    leaf_size = cols[2].slider('3. Leaf size', 20, 50, 30,
                            help="Leaf size passed to BallTree or KDTree. The optimal value depends on the nature of the problem.")
    cols1 = st.columns(3)
    metric = cols1[0].selectbox('4. Metric to use for distance computation', ['minkowski', 'manhattan', 'cosine', 'haversine'],
                            help="select one distance computation from provided metrics")
    set_model_parameters = {
        'n_neighbors': n_neighbors,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'metric': metric,
    }
    return set_model_parameters


def rdf_regression_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. The number of trees in the forest', 100, 1000, 100,
                        help="The greater the number of trees, the more time-consuming it becomes.")
    criterion = cols[1].selectbox('2. Criterion',
                         ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                         help="The function to measure the quality of a split.")
    max_depth = cols[2].selectbox('3. The maximum depth of the tree', [3, None, 2, 4, 5, 6, 7, 8, 9, 10],
                        help="If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    cols1 = st.columns(3)
    min_samples_split = cols1[0].selectbox('4. min_samples_split', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help="The minimum number of samples required to split an internal node.")
    min_samples_leaf = cols1[1].selectbox('5. min_samples_leaf', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help='The minimum number of samples required to be at a leaf node.')
    max_features = cols1[2].selectbox('6. max_features', ['sqrt', 'log2', 0.8, 0.9, 1.0],
                                          help='The number of features to consider when looking for the best split. If float, then draw % percentage of features.')
    set_model_parameters = {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': True,
        'random_state': 42,
        # 'n_jobs': n_jobs,
    }
    return set_model_parameters


def logisticregression_class_set_params(ml_method):
    cols = st.columns(3)
    penalty = cols[0].selectbox('1. the norm of the penalty', ['l2', 'l1','elasticnet', 'none'],
                            help="""
                            Specify the norm of the penalty:
                            **'none'**: no penalty is added;
                            **'l2'**: add a L2 penalty term and it is the default choice;
                            **'l1'**: add a L1 penalty term;
                            **'elasticnet'**: both L1 and L2 penalty terms are added.
                            """)
    tol = cols[1].selectbox('2. the tolerance for the optimization',
                         [1e-4, 2, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-6],
                         help="The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol, see Notes below.")
    C = cols[2].number_input('3. regularization', 0.0, 10000.0, 1.0,
                        help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.")
    cols1 = st.columns(3)
    random_state = cols1[0].slider('4. Controls the generation of the random states', 0, 137, 42,
                              help="""Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data  """)
    max_iter = cols1[1].slider('5. the maximum number of iterations', 100, 10000, 1000)
    l1_ratio = cols1[2].slider('6. l1 ratio', 0.0, 1.0, 0.0, help="""
                    In [scikit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), l1_ratio is the ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
                    """)
    set_model_parameters = {
        'penalty': penalty,
        'tol': tol,
        'C': C,
        'solver': 'saga',
        'random_state': random_state,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio,
        # 'n_jobs': n_jobs,
    }
    return set_model_parameters


def GaussianNB_class_set_params(ml_method):
    var_smoothing = st.selectbox('1. var_smoothing', [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                            help="Portion of the largest variance of all features that is added to variances for calculation stability.")
    set_model_parameters = {
        'var_smoothing': var_smoothing,
    }
    return set_model_parameters


def knn_class_set_params(ml_method):
    cols = st.columns(3)
    n_neighbors = cols[0].slider('1. Number of neighbors', 4, 30, 5,
                            help="")
    algorithm = cols[1].selectbox('2. Algorithm used to compute the nearest neighbors', ["auto", "ball_tree", "kd_tree", "brute"],
                            help="‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method. Note: fitting on sparse input will override the setting of this parameter, using brute force.")
    leaf_size = cols[2].slider('3. Leaf size', 20, 50, 30,
                            help="Leaf size passed to BallTree or KDTree. The optimal value depends on the nature of the problem.")
    cols1 = st.columns(3)
    metric = cols1[0].selectbox('4. Metric to use for distance computation', ['minkowski', 'manhattan', 'cosine', 'haversine'],
                            help="select one distance computation from provided metrics")
    set_model_parameters = {
        'n_neighbors': n_neighbors,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'metric': metric,
    }
    return set_model_parameters


def svm_class_set_params(ml_method):
    cols = st.columns(3)
    C = cols[0].selectbox('1. regularization strength', [1, 1e+5, 1e+4, 1e+1, 1e+3, 1e+2, 1e-1, 1e-2, 1e-3, 1e-4,1e-5,1e-6],
                        help="The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
    kernel = cols[1].selectbox('2. kernel function',
                         ['linear', 'poly', 'rbf', 'sigmoid'],
                         help="Specifies the kernel type to be used in the algorithm.")
    degree = cols[2].number_input('3. degree', 1, 10, 3,
                        help="Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.")
    cols1 = st.columns(3)
    gamma = cols1[0].selectbox('4. gamma', ['scale', 'auto', 1e-1, 1e-2, 1e-3, 1e-4,1e-5,1e-6], help="Kernel coefficient for rbf, poly and sigmoid")
    tol = cols1[1].selectbox('5. tolerance for stopping criterion', [1e-3, 1e-2, 1e-4,1e-5,1e-6])
    set_model_parameters = {
        'C': C,
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'tol': tol,
        'probability': True,
    }
    return set_model_parameters


def rdf_class_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. The number of trees in the forest', 100, 1000, 100,
                        help="The greater the number of trees, the more time-consuming it becomes.")
    criterion = cols[1].selectbox('2. criterion',
                         ['gini', 'entropy', 'log_loss'],
                         help="The function to measure the quality of a split.")
    max_depth = cols[2].selectbox('3. max_depth', [3, None, 2, 4, 5, 6, 7, 8, 9, 10],
                        help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    cols1 = st.columns(3)
    min_samples_split = cols1[0].selectbox('4. min_samples_split', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help="The minimum number of samples required to split an internal node.")
    min_samples_leaf = cols1[1].selectbox('5. min_samples_leaf', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help='The minimum number of samples required to be at a leaf node.')
    max_features = cols1[2].selectbox('6. max_features', ['sqrt', 'log2', None],
                                          help='The number of features to consider when looking for the best split.')
    set_model_parameters = {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': True,
        'random_state': 42,
    }
    return set_model_parameters


def adaboost_class_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. The number of decision trees', 100, 1000, 100,
                        help="A decision tree classifier is default estimator for adaboost. The greater the number of trees, the more time-consuming it becomes.")
    learning_rate = cols[1].selectbox('2. learning_rate', [1, 1e-4, 1e-3, 1e-2, 1e-1, 1e+1, 1e+2],
                         help="Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.")
    criterion = cols[2].selectbox('3. criterion', ['gini', 'entropy', 'log_loss'],
                                  help="The function to measure the quality of a split.")
    cols1 = st.columns(3)
    max_depth = cols1[0].selectbox('4. max_depth', [3, None, 2, 4, 5, 6, 7, 8, 9, 10],
                                  help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    min_samples_split = cols1[1].selectbox('5. min_samples_split', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help="The minimum number of samples required to split an internal node.")
    min_samples_leaf = cols1[2].selectbox('6. min_samples_leaf', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help='The minimum number of samples required to be at a leaf node.')
    cols2 = st.columns(3)
    max_features = cols2[0].selectbox('7. max_features', ['sqrt', 'log2', None],
                                          help='The number of features to consider when looking for the best split.')

    adaboost_body_parameters = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'random_state': 42,
    }
    decisiontree_model_paramters = {
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
    }
    set_model_parameters = {
        "adaboost_body_parameters": adaboost_body_parameters,
        "decisiontree_model_paramters": decisiontree_model_paramters,
                            }
    return set_model_parameters


def gradientboost_class_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. The number of boosting stages to perform.', 100, 1000, 100,
                        help="The greater the number of trees, the more time-consuming it becomes.")
    learning_rate = cols[1].selectbox('2. learning_rate', [1e-1, 1e-4, 1e-3, 1e-2, 1, 1e+1, 1e+2],
                                      help="Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.")
    criterion = cols[2].selectbox('3. criterion', ["friedman_mse", "squared_error"],
                                  help="The function to measure the quality of a split. ")

    cols1 = st.columns(3)
    max_depth = cols1[0].selectbox('4. max_depth', [3, None, 1, 2, 4, 5, 6, 7, 8, 9, 10],
                                  help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    min_samples_split = cols1[1].selectbox('5. min_samples_split', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help="The minimum number of samples required to split an internal node.")
    min_samples_leaf = cols1[2].selectbox('6. min_samples_leaf', [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help='The minimum number of samples required to be at a leaf node.')
    cols2 = st.columns(3)
    max_features = cols2[0].selectbox('7. max_features', ['sqrt', 'log2', None],
                                          help='The number of features to consider when looking for the best split.')
    subsample = cols2[1].slider('7. subsample', 0.5, 1.0, 1.0,
                                      help='The fraction of samples to be used for fitting the individual base learners.')

    set_model_parameters = {
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'subsample': subsample,
        'random_state': 42,
    }
    return set_model_parameters


def xgboost_class_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. The number of boosting stages to perform.', 100, 1000, 100,
                        help="The greater the number of trees, the more time-consuming it becomes.")
    learning_rate = cols[1].slider('2. learning_rate', 0.0, 1.0, 0.3,
                                      help="Step size shrinkage used in update to prevents overfitting.")
    gamma = cols[2].slider('3. gamma', 0, 100, 0,
                                  help="Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.")

    cols1 = st.columns(3)
    max_depth = cols1[0].slider('4. max_depth', 2, 20, 6,
                                  help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.")

    min_child_weight = cols1[1].slider('5. min_child_weight', 0, 100, 1, help="Minimum sum of instance weight (hessian) needed in a child. ")
    subsample = cols1[2].slider('6. subsample', 0.5, 1.0, 1.0,
                                help='Subsample ratio of the training instances. ')

    cols2 = st.columns(3)
    colsample_bytree = cols2[0].slider('7. colsample_bytree', 0.5, 1.0, 1.0,
                                help='The ratio of columns when constructing each tree.')
    early_stopping_rounds = cols2[1].selectbox('8. early_stopping_rounds', [20, 5, 10, 30, 40, 50],
                                          help='The model will train until the validation score stops improving. Validation error needs to decrease at least every early_stopping_rounds to continue training.')


    set_model_parameters = {
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        'gamma': gamma,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'early_stopping_rounds': early_stopping_rounds,
        'random_state': 42,
        'eval_metric': 'auc',
    }
    return set_model_parameters


def lightgbm_class_set_params(ml_method):
    cols = st.columns(3)
    n_estimators = cols[0].slider('1. Number of boosted trees to fit', 100, 1000, 100,
                        help="The greater the number of trees, the more time-consuming it becomes.")
    learning_rate = cols[1].slider('2. learning_rate', 0.0, 1.0, 0.1,
                                      help="Step size shrinkage used in update to prevents overfitting.")
    num_leaves = cols[2].slider('3. num_leaves', 0, 100, 31,
                                  help="Maximum tree leaves for base learners.")

    cols1 = st.columns(3)
    max_depth = cols1[0].slider('4. max_depth', 2, 20, 6,
                                  help="Maximum tree depth for base learners, <=0 means no limit.")
    subsample = cols1[1].slider('5. subsample', 0.5, 1.0, 1.0,
                                help='Subsample ratio of the training instances. ')
    colsample_bytree = cols1[2].slider('6. colsample_bytree', 0.5, 1.0, 1.0,
                                       help='Subsample ratio of columns when constructing each tree.')

    cols2 = st.columns(3)

    min_child_weight = cols2[0].selectbox('7. min_child_weight', [1e-3, 1e-4, 1e-2, 1e-1, 1],
                                       help="Minimum sum of instance weight (Hessian) needed in a child (leaf). ")
    min_child_samples = cols2[1].slider('8. min_child_samples', 1, 30, 20,
                                          help='Minimum number of data needed in a child (leaf).')


    set_model_parameters = {
        'n_estimators': n_estimators,
        "learning_rate": learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_samples': min_child_samples,
        'random_state': 42,
        'n_jobs': 1,
    }
    return set_model_parameters


def catboost_class_set_params(ml_method):
    cols = st.columns(3)
    iterations = cols[0].slider('1. The maximum number of trees', 100, 1000, 100,
                        help="The maximum number of trees that can be built when solving machine learning problems.")
    learning_rate = cols[1].slider('2. The learning rate', 0.0, 1.0, 0.01,
                                      help="Used for reducing the gradient step.")
    max_leaves = cols[2].slider('3. The maximum number of leafs in the resulting tree', 0, 50, 31,
                                  help="The maximum number of leafs in the resulting tree. ")

    cols1 = st.columns(3)
    depth = cols1[0].slider('4. Depth of the tree', 0, 10, 6,
                                  help="Maximum tree depth for base learners, <=0 means no limit.")
    rsm = cols1[1].slider('6. colsample_bylevel', 0.5, 1.0, 1.0,
                          help='Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.')
    min_data_in_leaf = cols1[2].slider('8. min_data_in_leaf', 1, 20, 1,
                                          help='The minimum number of training samples in a leaf.')
    set_model_parameters = {
        # 'loss_function': "Logloss",
        'iterations': iterations,
        "learning_rate": learning_rate,
        'max_leaves': max_leaves,
        'depth': depth,
        'rsm': rsm,
        'min_data_in_leaf': min_data_in_leaf,
        "eval_metric": 'AUC',
    }
    return set_model_parameters


def ml_model_set_paras(ml_method):
    if ml_method == 'Linear Regression':
        set_model_parameters = linear_reg_set_params(ml_method)
        model = LinearRegression(**set_model_parameters)
    elif ml_method == 'Lasso Regression':
        set_model_parameters = lasso_reg_set_params(ml_method)
        model = Lasso(**set_model_parameters)
    elif ml_method == 'Ridge Regression':
        set_model_parameters = ridge_reg_set_params(ml_method)
        model = Ridge(**set_model_parameters)
    elif ml_method == 'ElasticNet Regression':
        set_model_parameters = elasticnet_reg_set_params(ml_method)
        model = ElasticNet(**set_model_parameters)
    elif ml_method == 'Bayesian Ridge Regression':
        set_model_parameters = bayesian_ridge_set_params(ml_method)
        model = BayesianRidge(**set_model_parameters)
    elif ml_method == "Support Vector Regression":
        set_model_parameters = svr_set_params(ml_method)
        model = SVR(**set_model_parameters)
    elif ml_method == "K-Nearest Neighbors Regression":
        set_model_parameters = knn_regression_set_params(ml_method)
        model = KNeighborsRegressor(**set_model_parameters, n_jobs=n_jobs)
    elif ml_method == "Random Forest Regressor":
        set_model_parameters = rdf_regression_set_params(ml_method)
        model = RandomForestRegressor(**set_model_parameters, n_jobs=n_jobs)
    elif ml_method == 'Logistic Regression':
        set_model_parameters = logisticregression_class_set_params(ml_method)
        model = LogisticRegression(**set_model_parameters, n_jobs=n_jobs)
    elif ml_method == 'Random Forest Classification':
        set_model_parameters = rdf_class_set_params(ml_method)
        model = RandomForestClassifier(**set_model_parameters, n_jobs=n_jobs)
    elif ml_method == 'Support Vector Classification':
        set_model_parameters = svm_class_set_params(ml_method)
        model = SVC(**set_model_parameters)
    elif ml_method == 'AdaBoost':
        set_model_parameters = adaboost_class_set_params(ml_method)
        model = AdaBoostClassifier(**set_model_parameters["adaboost_body_parameters"],
                                   estimator=DecisionTreeClassifier(**set_model_parameters["decisiontree_model_paramters"]))
    elif ml_method == 'GradientBoosting':
        set_model_parameters = gradientboost_class_set_params(ml_method)
        model = GradientBoostingClassifier(**set_model_parameters)
    elif ml_method == 'Xgboost':
        set_model_parameters = xgboost_class_set_params(ml_method)
        model = XGBClassifier(**set_model_parameters, n_jobs=n_jobs)
    elif ml_method == 'LightGBM':
        set_model_parameters = lightgbm_class_set_params(ml_method)
        model = LGBMClassifier(**set_model_parameters)
    elif ml_method == "CatBoost":
        set_model_parameters = catboost_class_set_params(ml_method)
        model = CatBoostClassifier(**set_model_parameters, thread_count=n_jobs)
    elif ml_method == "Gaussian Naive Bayes":
        set_model_parameters = GaussianNB_class_set_params(ml_method)
        model = GaussianNB(**set_model_parameters)
    elif ml_method == "K-Nearest Neighbors":
        set_model_parameters = knn_class_set_params(ml_method)
        model = KNeighborsClassifier(**set_model_parameters, n_jobs=n_jobs)
    else:
        set_model_parameters = None
        model = None
    return set_model_parameters, model