from sklearn.model_selection import StratifiedKFold
import numpy as np

import model.question_loader as ql
import evaluation.shared as shared

clfs = [shared.SVM, shared.LOGREG, shared.LDA, shared.MNB]
question_loader = ql.QuestionLoader(
    qfile=shared.QFILE, catfile=shared.CATFILE, subcats=shared.SUBCATS,
    metadata=True, verbose=True)

skf = StratifiedKFold(n_splits=shared.CV, random_state=42)

questions = np.asarray(question_loader.questions)
categoryids = np.asarray(question_loader.categoryids)
corr_micro = []
corr_macro = []
for train_index, test_index in skf.split(questions, categoryids):
    q_train, q_test = questions[train_index], questions[test_index]
    cat_train, cat_test = categoryids[train_index], categoryids[test_index]

    # fit all classifiers
    for clf in clfs:
        clf.fit(q_train, cat_train)
    # predict_proba for all classfifiers with best param config
    probas = [clf.predict_proba(q_test) for clf in clfs]

    # micro-averaged cov
    probas_micro = np.asarray([prob.reshape(-1) for prob in probas])
    corr_micro.append(np.cov(probas_micro))

    # macro-averaged cov
    probas_macro = np.asarray(probas)
    corr_macro_class = [np.cov(probs) for probs in np.rollaxis(probas_macro,
                                                               2)]
    corr_macro.append(np.mean(np.asarray(corr_macro_class), 0))

corr_micro = np.mean(np.asarray(corr_micro), axis=0)
corr_macro = np.mean(np.asarray(corr_macro), axis=0)
shared.save_and_report(
    results={'corr_micro': corr_micro, 'corr_macro': corr_macro},
    folder='ensemble', name='corr')
