"""
The :mod: `shared` implements some constants and
helper functions for performing and saving the gridsearch results
"""
# Author: Ingo Guehring

from time import gmtime, strftime
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from model import SMSGuruModel


# number of folds used in cross-validation
CV = 5
# number of folds used in cross-validation in probability calibration
# CCV = 2
CCV = KFold(n_splits=3)

QFILE = './data/question_train.csv'
CATFILE = './data/category.csv'

# in milestone it suffices to focus on parent categories
SUBCATS = True

# number of jobs used in CV -> parallelize
N_JOBS = -1
N_PARENTCATS = 14

# deprecated
# # for SVM RBF
# C_RANGE = np.logspace(-2, 6, 9)
# GAMMA_RANGE = np.logspace(-8, 0, 9)

# for SVM RBF
# see time plot
# C_RANGE = np.logspace(-2, 5, 8)
# GAMMA_RANGE = np.logspace(-7, 0, 8)

GAMMA_RANGE = np.logspace(-3, 3, 7)
C_RANGE = np.logspace(-3, 3, 7)

# this choice is based on [Seb99]
MIN_DF = [2, 3]

# define classifiers here

# use isotonic calibration
MNB_subcats = SMSGuruModel(
    CalibratedClassifierCV(MultinomialNB(), method='isotonic', cv=CCV),
    metadata=False, memory=True).model

MNB_parentcats = SMSGuruModel(
    CalibratedClassifierCV(MultinomialNB(), method='isotonic'),
    metadata=False, memory=True).model

# use sigmoid calibration
SVM_subcats = SMSGuruModel(
    classifier=CalibratedClassifierCV(LinearSVC(C=0.1), cv=CCV),
    metadata=False).model

SVM_parentcats = SMSGuruModel(
    classifier=CalibratedClassifierCV(LinearSVC(C=0.1)),
    metadata=False).model

# no info about calibration
LDA = SMSGuruModel(classifier=LDA(), min_df=3, memory=True).model

# no calibration necessary
LOGREG = SMSGuruModel(
    classifier=LogisticRegression(C=10)).model

# use sigmoid caibration
RF = SMSGuruModel(
    CalibratedClassifierCV(RandomForestClassifier(n_estimators=500), cv=CCV),
    metadata=False).model


def merge_two_dicts(x, y):
    """Return merged dict"""
    z = x.copy()
    z.update(y)
    return z


def save_and_report(results, folder, name='grids_cv.npy', report=None):
    """Save results and info about the performed search"""

    # save results
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    path = './results/gridsearch/' + folder + '/'
    filename = current_time + 'f1_macro_' + name
    np.save(path + filename, results)

    if report is not None:
        # update report
        with open(path + "/gridsearches.txt", "a") as report:
            report.write("performed at: {}, non_grid_params:  "
                         .format(current_time) + report)
