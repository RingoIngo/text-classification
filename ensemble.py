"""
The :mod: `ensemble` combines multiple classifiers in a voting schema
"""
# Author: Ingo Guehring

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score

from model import SMSGuruModel
import model.question_loader as ql
import evaluation.shared as shared

# define classifiers here
mnb = SMSGuruModel(classifier=MultinomialNB(), reduction=None,
                   metadata=False, memory=True).model
svm = SMSGuruModel(classifier=LinearSVC(C=0.1), reduction=None).model
lda = SMSGuruModel(classifier=LDA(), reduction=None, memory=True).model

ensemble = VotingClassifier(
    estimators=[('mnb', mnb), ('svm', svm), ('lda', lda)], voting='soft')

qfile = './data/question_train.csv'
catfile = './data/category.csv'
subcats = False

question_loader = ql.QuestionLoader(
    qfile=qfile, catfile=catfile, subcats=subcats, metadata=True, verbose=True)

scores = cross_val_score(
    ensemble, question_loader.questions, question_loader.categoryids, cv=5,
    scoring='f1_macro', n_jobs=-1, verbose=100)

shared.save_and_report(
    results=scores, folder='ensemble', name='gen_error.npy')
