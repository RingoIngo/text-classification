from time import gmtime, strftime
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

import model
import evaluation.shared as shared

MODEL = model.SMSGuruModel(classifier=LDA(n_components=2),
                           reduction=None, memory=True)
MODEL.set_question_loader(subcats=shared.SUBCATS)

skf = StratifiedKFold(n_splits=5)
X = MODEL.question_loader_.questions
y = MODEL.question_loader_.categoryids
categories = MODEL.question_loader_.categories
projected_test = None
i = 0
for train_index, test_index in skf.split(X, y):
    if i == 0:
        X_test = np.array(X)[test_index]
        y_test = np.array(y)[test_index]
        X_train = np.array(X)[train_index]
        y_train = np.array(y)[train_index]

        MODEL.model.fit(X_train, y_train)
        projected_test = MODEL.model.transform(X_test)
    i = 1

folder = 'lda/plot'
name = '2dim_plot'
current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
path = './results/gridsearch/' + folder + '/'
filename = current_time + name
np.savez(path + filename, projected_test, y_test, categories)
