"""
The :mod: `evaluate_lda_svm` implements the model and the constants needed
for the evalutation of LDA as dimensionality reduction and SVM as
classifier"""
# Author: Ingo Gühring

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import evaluation
import smsguru_model as sms


sms_guru_model_lda_svm = sms.SMSGuruModel(classifier=SVC, reduction=LDA)

# grid
n_components_range = np.arange(1, evaluation.n_parentcats + 1)
# kernels = ['linear', 'rbf']
gamma_range = np.logspace(-3, 3, 10)
C_range = np.logspace(-3, 3, 10)

if __name__ == "__main__":
    grid_gridsearch = [dict(reduce_dim__n_components=n_components_range,
                            # classifier_kernel=kernels,
                            classifier_gamma=gamma_range,
                            classifier_C=C_range)]

    sms_guru_model_lda_svm.set_question_loader(subcats=evaluation.subcats)
    sms_guru_model_lda_svm.gridsearch(param_grid=grid_gridsearch,
                                      n_jobs=evaluation.n_jobs,
                                      CV=evaluation.CV)
    evaluation.save_and_report(
        results=sms_guru_model_lda_svm.grid_search_.cv_results_,
        folder='lda_svm')

    # since in this case the higher the dimension the better the estimator
    # we do not include the lower dimensions in this search
    grid_generalization_error = [dict(classifier_gamma=gamma_range,
                                      classifier_C=C_range)]
    nested_scores = sms_guru_model_lda_svm.nested_cv(
        param_grid=grid_generalization_error, CV=evaluation.CV)
    evaluation.save_and_report(results=nested_scores,
                               folder='lda_svm',
                               name='gen_error.npy')
