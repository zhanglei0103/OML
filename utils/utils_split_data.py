import sys
import os

from sklearn.model_selection import (LeaveOneOut, LeavePOut, RepeatedStratifiedKFold, RepeatedKFold, train_test_split)


def select_cross_validation(n_splits, n_repeats, cv_method, random_state=42, leavep=2):
    if cv_method == 'RepeatedStratifiedKFold':
         cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    elif cv_method == 'RepeatedKFold':
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    elif cv_method == "LeaveOneOut":
        cv = LeaveOneOut()
    elif cv_method == "LeavePOut":
        cv = LeavePOut(p=leavep)
    else:
        raise NotImplementedError(f'Cross validation method ({cv_method}) not implemented')
    return cv
