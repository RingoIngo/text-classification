import smsguru_model as sms
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from time import gmtime, strftime
import numpy as np

if __name__ == "__main__":
    def merge_two_dicts(x, y):
        """Return merged dict"""
        z = x.copy()
        z.update(y)
        return z

    classifier = MultinomialNB()
    subcats = False
    metadata = False
    # the dimensions used in the dim reduction step
    N_DIM_OPTIONS = [10, 30, 60, 100, 200, 500, 1000, 1500, 2500, 5000]

    # this choice is based on [Seb99]
    MIN_DF = [1, 2, 3]

    base_grid = (
        dict(union__question_bow__tokens__mapdates=[True, False],
             union__question_bow__tokens__mapnumbers=[True, False],
             union__question_bow__tokens__spellcorrect=[True, False],
             union__question_bow__tokens__stem=[True, False],
             # union__question_bow__tokens__tokenizer=['word_tokenizer',
             #                                         'word_punct_tokenizer'],
             union__question_bow__tokens__tokenizer=['word_punct_tokenizer'],
             # union__question_bow__vectorize__binary=[True, False],
             union__question_bow__vectorize__min_df=MIN_DF,
             union__question_bow__tfidf=[None, TfidfTransformer()],
             )
    )

    univariate = (
        dict(union__question_bow__reduce_dim=[SelectKBest(chi2)],
             union__question_bow__reduce_dim__k=N_DIM_OPTIONS)
    )

    multivariate = (
        dict(union__question_bow__reduce_dim=[TruncatedSVD()],
             union__question_bow__reduce_dim__n_components=N_DIM_OPTIONS)
    )

    grid = [merge_two_dicts(base_grid, univariate),
            merge_two_dicts(base_grid, multivariate)]

    # TEST
    test_param_grid = [
        dict(union__question_bow__tokens__mapdates=[False],
             union__question_bow__tokens__mapnumbers=[False],
             union__question_bow__tokens__spellcorrect=[False],
             union__question_bow__tokens__stem=[False],
             union__question_bow__tokens__tokenizer=['word_punct_tokenizer'],
             union__question_bow__vectorize__binary=[False],
             union__question_bow__vectorize__min_df=[1, 2],
             union__question_bow__tfidf=[None],
             union__question_bow__reduce_dim=[SelectKBest(chi2)],
             union__question_bow__reduce_dim__k=[2000],
             ),
    ]

    sms_guru_model = (sms.SMSGuruModel(classifier=classifier, metadata=metadata).
                      set_question_loader(subcats=subcats).
                      gridsearch(param_grid=test_param_grid,n_jobs=-1))
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            sms_guru_model.grid_search_.cv_results_)
    with open("./results/gridsearch/gridsearches.txt", "a") as report:
        report.write("""performed at: {}, non_grid_params:  subcats: {},
                      metadata: {}, classifier: MultinomialNB\n"""
                     .format(current_time, subcats, metadata))
