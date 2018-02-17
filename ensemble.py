"""
The :mod: `ensemble` combines multiple classifiers in a voting schema
"""
# Author: Ingo Guehring

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from model import SMSGuruModel
import model.question_loader as ql
import evaluation.shared as shared

# define classifiers here
mnb = SMSGuruModel(classifier=MultinomialNB(), reduction=None,
                   metadata=False, memory=True).model
svm = SMSGuruModel(
    classifier=CalibratedClassifierCV(LinearSVC(C=0.1)), reduction=None).model
lda = SMSGuruModel(classifier=LDA(), reduction=None, memory=True).model
logreg = SMSGuruModel(
    classifier=LogisticRegression(C=10), reduction=None).model

ensemble = VotingClassifier(
    estimators=[('mnb', mnb), ('logreg', logreg), ('lda', lda)], voting='soft')

qfile = './data/question_train.csv'
catfile = './data/category.csv'
subcats = False
cv = 5
verbose = 100
question_loader = ql.QuestionLoader(
    qfile=qfile, catfile=catfile, subcats=subcats, metadata=True, verbose=True)

# ##################### without gridsearch ###############################
# scores = cross_val_score(
#     ensemble, question_loader.questions, question_loader.categoryids, cv=cv,
#     scoring='f1_macro', n_jobs=-1, verbose=verbose)
#
# shared.save_and_report(
#     results=scores, folder='ensemble', name='gen_error.npy')

# ##################### with gridsearch ###############################
# svm param
C_RANGE = np.logspace(-5, 5, 11)

# grid
PARAM_GRID = {'logreg__classifier__C': C_RANGE}

grid = GridSearchCV(
    estimator=ensemble, cv=cv, param_grid=PARAM_GRID, scoring='f1_macro',
    refit=False, error_score=-1, n_jobs=-1, verbose=verbose)

grid.fit(question_loader.questions, question_loader.categoryids)
print(grid.cv_results_)
shared.save_and_report(results=grid.cv_results_, folder='ensemble')


clf = GridSearchCV(estimator=ensemble, param_grid=PARAM_GRID, cv=cv,
                   n_jobs=-1, scoring='f1_macro', verbose=verbose)
nested_cv_scores = cross_val_score(
    clf, X=question_loader.questions, y=question_loader.categoryids, cv=cv,
    scoring='f1_macro', verbose=verbose)
shared.save_and_report(
    results=nested_cv_scores, folder='ensemble', name='gen_error.npy')
