from functools import reduce
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
    subcats = True
    metadata = False
    CV = 3
    # the dimensions used in the dim reduction step
    N_DIM_OPTIONS = [50, 100, 200, 500, 1000, 1500, 2500]

    # this choice is based on [Seb99]
    MIN_DF = [1, 2, 3]

    base_grid = (
        dict(union__bow__tokens__spellcorrect=[True, False],
             union__bow__tokens__stem=[True, False],
             # union__bow__tokens__tokenizer=['word_tokenizer',
             #                                         'word_punct_tokenizer'],
             union__bow__tokens__tokenizer=['word_punct_tokenizer'],
             # union__bow__vectorize__binary=[True, False],
             union__bow__vectorize__min_df=MIN_DF,
             union__bow__tfidf=[None, TfidfTransformer()],
             )
    )

    mapnumerics = (
        dict(union__bow__tokens__mapdates=[True],
             union__bow__tokens__mapnumbers=[True])
    )


    no_mapnumerics = (
        dict(union__bow__tokens__mapdates=[False],
             union__bow__tokens__mapnumbers=[False])
    )

    univariate = (
        dict(union__bow__reduce_dim=[SelectKBest(chi2)],
             union__bow__reduce_dim__k=N_DIM_OPTIONS)
    )

    multivariate = (
        dict(union__bow__reduce_dim=[TruncatedSVD()],
             union__bow__reduce_dim__n_components=N_DIM_OPTIONS)
    )

    grid = [reduce(merge_two_dicts, [base_grid, mapnumerics, univariate]),
            reduce(merge_two_dicts, [base_grid, no_mapnumerics, multivariate])]

    # TEST
    test_param_grid = [
        dict(union__bow__tokens__mapdates=[True],
             union__bow__tokens__mapnumbers=[True],
             union__bow__tokens__spellcorrect=[False],
             union__bow__tokens__stem=[False],
             union__bow__tokens__tokenizer=['word_punct_tokenizer'],
             union__bow__vectorize__binary=[False],
             union__bow__vectorize__min_df=[3],
             union__bow__tfidf=[None],
             union__bow__reduce_dim=[SelectKBest(chi2)],
             union__bow__reduce_dim__k=[8000],
             ),
    ]

    sms_guru_model = (sms.SMSGuruModel(classifier=classifier, metadata=metadata).
                      set_question_loader(subcats=subcats).
                      gridsearch(param_grid=grid, n_jobs=-1, CV=CV))
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            sms_guru_model.grid_search_.cv_results_)
    with open("./results/gridsearch/gridsearches.txt", "a") as report:
        report.write("""performed at: {}, non_grid_params:  subcats: {},
                      metadata: {}, CV: {}, classifier: MultinomialNB\n"""
                     .format(current_time, subcats, metadata, CV))
