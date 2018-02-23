""" The :mod: `ensemble` combines multiple classifiers in a voting schema """
# Author: Ingo Guehring

# from sklearn.ensemble import VotingClassifier
import argparse
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

import model.question_loader as ql
from model.ensemble import GridSearchCVB, VotingClassifierB, f1_macroB
import evaluation.shared as shared


def evaluate(subcats=False, comb_method='avg',
             save_avg_path='./results/gridsearch/ensemble/raw/'):
    print('subcats: {}, comb_method: {}'
          ', save_avg_path: {}'.format(
                  subcats, comb_method, save_avg_path))

    if not os.path.exists(save_avg_path):
        print('create directory: {}'.format(save_avg_path))
        os.makedirs(save_avg_path)

    question_loader = ql.QuestionLoader(
        qfile=shared.QFILE, catfile=shared.CATFILE, subcats=subcats,
        metadata=True, verbose=True)

    cv = 5
    verbose = 100

    if comb_method != 'bagging':
        print(comb_method)
        # If a classifier is changed the grid might have to be changed, too
        # Put the estimator with the best expected perfromance at the first
        # position! Then its probability output will be saved!
        ensemble = VotingClassifierB(
            estimators=[('svm', shared.SVM),
                        ('mnb', shared.MNB),
                        ('lda', shared.LDA)], voting='soft',
            comb_method=comb_method, save_avg_path=save_avg_path)

        # ##################### without gridsearch ############################
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
        PARAM_GRID = {'svm__classifier__base_estimator__C': C_RANGE,
                      'svm__union__bow__tokens__stem': [True, False],
                      'svm__union__bow__vectorize__min_df': shared.MIN_DF,
                      'svm__union__bow__tfidf': [None, TfidfTransformer()],
                      'mnb__union__bow__tokens__stem': [True, False],
                      'mnb__union__bow__vectorize__min_df': shared.MIN_DF,
                      'mnb__union__bow__tfidf': [None, TfidfTransformer()],
                      'lda__union__bow__tokens__stem': [True, False],
                      'lda__union__bow__vectorize__min_df': shared.MIN_DF,
                      'lda__union__bow__tfidf': [None, TfidfTransformer()]}

        # grid = GridSearchCV(
        #     estimator=ensemble, cv=cv, param_grid=PARAM_GRID,
        #     refit=False, error_score=-1, n_jobs=-1, verbose=verbose)
        #
        # grid.fit(question_loader.questions, question_loader.categoryids)
        # print(grid.cv_results_)
        # shared.save_and_report(results=grid.cv_results_, folder='ensemble')

        clf = GridSearchCVB(estimator=ensemble, param_grid=PARAM_GRID, cv=cv,
                            n_jobs=-1, scoring='f1_macro', verbose=verbose)

        nested_cv_scores = cross_val_score(
            clf, X=question_loader.questions, y=question_loader.categoryids,
            cv=cv, scoring=f1_macroB, verbose=verbose)

    if comb_method == 'bagging':
        base_estimator = shared.SVM
        base_estimator.set_params(
            question_created_at=None,
            union__bow__selector=None)

        clf = BaggingClassifier(
            base_estimator, n_estimators=50, max_samples=1.0)
        clf

        X = [pair['question'] for pair in question_loader.questions]
        X = np.asarray(X).reshape((-1, 1))
        nested_cv_scores = cross_val_score(
            clf, X=X,
            y=question_loader.categoryids, cv=cv, scoring=f1_macroB,
            verbose=verbose)

    shared.save_and_report(
        results=nested_cv_scores, folder='ensemble',
        name=comb_method + 'gen_error.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate ensemble')
    parser.add_argument('-cm', '--comb_method',
                        choices=['avg', 'mult', 'bagging'],
                        default=argparse.SUPPRESS)

    parser.add_argument('-sc', '--subcats', dest='subcats',
                        action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('--save_avg_path', dest='save_avg_path',
                        default=argparse.SUPPRESS)

    evaluate(**vars(parser.parse_args()))
