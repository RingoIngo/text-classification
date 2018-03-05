"""
The :mod: `predict_question_category` loads and applies fitted models
"""
# Author: Ingo Guehring

import pickle
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

import model.question_loader as ql
import evaluation.shared as shared


def predict_question_category(qfile='questions.csv'):
    """Load and apply fitted classifiers for parent and subcats problem

    The classifier that is used is a linear Support Vector Machine. More
    detailied properties of the classifier are described in the report for
    milestone three.

    Parameters
    -----------
    qfile : csv file containing the test questions

    Returns
    ---------
    prediction : list containing a dictionary for each test questions.
        The dictionary contains the following elements:
            - major_category: id of predicted main category
            - minor_category: id of predicted subcategory
            -  confidence_major_cat: confidence for main prediction
            -  confidence_minor_cat: confidence for minor prediction
    """
    # load the model from disk
    model_subcats = pickle.load(open('model_subcats.sav', 'rb'))
    model_parentcats = pickle.load(open('model_parentcats.sav', 'rb'))

    # Extract test questions
    # metadata=False: if true performance might be better, but questions that
    # don't have a creation date are excluded from the test set.
    question_loader = ql.QuestionLoader(metadata=False)
    question_loader.read_test_data(qfile)
    qtest = question_loader.test_questions

    # minor
    probas_subcats = model_subcats.predict_proba(qtest)
    subcats = model_subcats.classes_
    minor_categories = subcats[np.argmax(probas_subcats, axis=1)]
    confidence_minor_cats = np.max(probas_subcats, axis=1)

    # major
    probas_parentcats = model_parentcats.predict_proba(qtest)
    parentcats = model_parentcats.classes_
    major_categories = parentcats[np.argmax(probas_parentcats, axis=1)]
    confidence_major_cats = np.max(probas_parentcats, axis=1)

    n_samples = len(minor_categories)
    prediction = []
    for sample in np.arange(n_samples):
        prediction.append(
            {'major_category': major_categories[sample],
             'minor_category': minor_categories[sample],
             'confidence_major_cat': confidence_major_cats[sample],
             'confidence_minor_cat': confidence_minor_cats[sample]})

    return prediction


def fit_and_save_final_predictor():
    """
    Fit and save the fitted final prediction method for sub and parent cats
    """

    PARAM_GRID = {'classifier__base_estimator__C': shared.C_RANGE,
                  'union__bow__vectorize__min_df': shared.MIN_DF,
                  'union__bow__tfidf': [None, TfidfTransformer()]}

    # subcats
    question_loader = ql.QuestionLoader(
        qfile=shared.QFILE, catfile=shared.CATFILE, subcats=True,
        metadata=False, verbose=True)

    grid_subcats = GridSearchCV(
        estimator=shared.SVM_subcats, cv=shared.CV, param_grid=PARAM_GRID,
        refit=True, error_score=-1, n_jobs=-1, verbose=100,
        scoring='f1_macro')

    grid_subcats.fit(question_loader.questions, question_loader.categoryids)
    pickle.dump(grid_subcats.best_estimator_, open('model_subcats.sav', 'wb'))

    # parent cats
    question_loader = ql.QuestionLoader(
        qfile=shared.QFILE, catfile=shared.CATFILE, subcats=False,
        metadata=False, verbose=True)

    grid_parentcats = GridSearchCV(
        estimator=shared.SVM_parentcats, cv=shared.CV, param_grid=PARAM_GRID,
        refit=True, error_score=-1, n_jobs=-1, verbose=100,
        scoring='f1_macro')

    grid_parentcats.fit(question_loader.questions, question_loader.categoryids)
    pickle.dump(grid_parentcats.best_estimator_,
                open('model_parentcats.sav', 'wb'))


def fit_or_predict(fit=False):
    if fit:
        fit_and_save_final_predictor()
    else:
        predict_question_category()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fit or predict')
    parser.add_argument('--fit', dest='fit',
                        action='store_true',
                        default=argparse.SUPPRESS)

    fit_or_predict(**vars(parser.parse_args()))
