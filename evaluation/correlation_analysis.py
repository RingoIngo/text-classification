from sklearn.model_selection import StratifiedKFold
import numpy as np

import model.question_loader as ql
import evaluation.shared as shared
import ensemble as en

clfs = [en.svm, en.logreg, en.lda, en.mnb]
question_loader = ql.QuestionLoader(
    qfile=shared.QFILE, catfile=shared.CATFILE, subcats=shared.SUBCATS,
    metadata=True, verbose=True)

skf = StratifiedKFold(n_splits=shared.CV, random_state=42)

questions = question_loader.questions
categoryids = question_loader.categoryids
corr = []
for train_index, test_index in skf.split(questions, categoryids):
    q_train, q_test = questions[train_index], questions[test_index]
    cat_train, cat_test = categoryids[train_index], categoryids[test_index]

    # fit all classifiers
    map(lambda clf: clf.fit(q_train, cat_train), clfs)
    # predict_proba for all classfifiers with best param config
    probas = [clf.predict_proba(q_test).reshape(-1) for clf in clfs]
    probas = np.asarray(probas).T
    corr.append(np.cov(probas))
corr = np.mean(np.asarray(corr), axis=0)
shared.save_and_report(results=corr, folder='ensemble', name='corr')
