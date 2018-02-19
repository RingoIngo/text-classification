""" The :mod: `ensemble` combines multiple classifiers in a voting schema """
# Author: Ingo Guehring

# from sklearn.ensemble import VotingClassifier
import argparse
import os
import numpy as np

from sklearn.model_selection import cross_val_score

import model.question_loader as ql
from model.ensemble import GridSearchCVB, VotingClassifierB
import evaluation.shared as shared


def evaluate(subcats=False, comb_method='avg',
             save_avg_path='./results/gridsearch/ensemble/raw/'):
    print('subcats: {}, comb_method: {}'
          ', save_avg_path: {}'.format(
                  subcats, comb_method, save_avg_path))

    if not os.path.exists(save_avg_path):
        print('create directory: {}'.format(save_avg_path))
        os.makedirs(save_avg_path)

    # If a classifier is changed the grid might have to be changed, too
    ensemble = VotingClassifierB(
        estimators=[('mnb', shared.MNB),
                    ('svm', shared.SVM),
                    ('lda', shared.LDA)], voting='soft',
        comb_method=comb_method, save_avg_path=save_avg_path)

    cv = 5
    verbose = 100
    question_loader = ql.QuestionLoader(
        qfile=shared.QFILE, catfile=shared.CATFILE, subcats=subcats,
        metadata=True, verbose=True)

    # ##################### without gridsearch ###############################
    # scores = cross_val_score(
    #     ensemble, question_loader.questions,
    # question_loader.categoryids, cv=cv,
    #     scoring='f1_macro', n_jobs=-1, verbose=verbose)
    #
    # shared.save_and_report(
    #     results=scores, folder='ensemble', name='gen_error.npy')

    # ##################### with gridsearch ###############################
    # svm param
    C_RANGE = np.logspace(-5, 5, 11)

    # grid
    PARAM_GRID = {'svm__classifier__base_estimator__C': C_RANGE}

    # grid = GridSearchCV(
    #     estimator=ensemble, cv=cv, param_grid=PARAM_GRID, scoring='f1_macro',
    #     refit=False, error_score=-1, n_jobs=-1, verbose=verbose)
    #
    # grid.fit(question_loader.questions, question_loader.categoryids)
    # print(grid.cv_results_)
    # shared.save_and_report(results=grid.cv_results_, folder='ensemble')

    clf = GridSearchCVB(estimator=ensemble, param_grid=PARAM_GRID, cv=cv,
                        n_jobs=-1, scoring='f1_macro', verbose=verbose)
    nested_cv_scores = cross_val_score(
        clf, X=question_loader.questions, y=question_loader.categoryids, cv=cv,
        scoring='f1_macro', verbose=verbose)
    shared.save_and_report(
        results=nested_cv_scores, folder='ensemble', name='gen_error.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate ensemble')
    parser.add_argument('-cm', '--comb_method',
                        choices=['avg', 'mult'],
                        default=argparse.SUPPRESS)

    parser.add_argument('-sc', '--subcats', dest='subcats',
                        action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('--save_avg_path', dest='save_avg_path',
                        default=argparse.SUPPRESS)

    evaluate(**vars(parser.parse_args()))
