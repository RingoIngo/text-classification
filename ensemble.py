"""
The :mod: `ensemble` combines multiple classifiers in a voting schema
"""
# Author: Ingo Guehring

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

import model.question_loader as ql
import evaluation.shared as shared


ensemble = VotingClassifier(
    estimators=[('mnb', shared.MNB),
                ('logreg', shared.LOGREG),
                ('lda', shared.LDA)], voting='soft')

subcats = False
cv = 5
verbose = 100
question_loader = ql.QuestionLoader(
    qfile=shared.QFILE, catfile=shared.CATFILE, subcats=subcats,
    metadata=True, verbose=True)

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
