"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set.
"""
# Author: Ingo Gühring

from time import gmtime, strftime
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import text_tokenizer_and_cleaner as ttc
import question_loader as ql

# taken from scikit learn examples:
# "Feature Union with Heterogeneous Data Sources"
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = np.recarray(shape=(len(posts),),
                               dtype=[('subject', object), ('body', object)])
        for i, text in enumerate(posts):
            headers, _, bod = text.partition('\n\n')
            bod = strip_newsgroup_footer(bod)
            bod = strip_newsgroup_quoting(bod)
            features['body'][i] = bod

            prefix = 'Subject:'
            sub = ''
            for line in headers.split('\n'):
                if line.startswith(prefix):
                    sub = line[len(prefix):]
                    break
            features['subject'][i] = sub

        return features


def _identity(words):
    return words


def create_pipeline(estimator=None):
    """Chain processing steps

    Build a `Pipeline` in which all processing steps are chained.
    1. tokenize and clean text
    2. vectorize tokens
    3. tfidf weighting
    4. TruncatedSVD reduction
    5. estimator (optinal)

    Parameters
    ----------
    estimator : an estimator, optinal, which is applied as last step
        in the chain. Default is None.
    """

    steps = [
        ('tokens', ttc.TextTokenizerAndCleaner()),
        ('vectorize', CountVectorizer(tokenizer=_identity,
                                      preprocessor=None,
                                      lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', SelectKBest(chi2, k=500))
    ]

    if estimator is not None:
        # Add the estimator
        steps.append(('classifier', estimator))
    return Pipeline(steps)


def evaluate_model(qfile='question_train.csv',
                   catfile='category.csv',
                   subcats=True,
                   verbose=10):
    CV = 5
    # the dimensions used in the dim reduction step
    N_DIM_OPTIONS = [500]
    # for multiclass it holds:
    # recall_micro = precision_micro = f1_micro = accuracy
    SCORES = ['recall_macro', 'precision_macro', 'f1_macro', 'f1_micro']
    # this choice is based on [Seb99]
    MIN_DF = [1, 2, 3]

    # model = create_pipeline(KNeighborsClassifier())
    model = create_pipeline(MultinomialNB())
    # param_grid = [
    #     # multivariate feature selection
    #     dict(tokens__mapdates=[True, False],
    #          tokens__mapnumbers=[True, False],
    #          tokens__spellcorrect=[True, False],
    #          tokens__stem=[True, False],
    #          tokens__tokenizer=['word_tokenizer',
    #                             'word_punct_tokenizer'],
    #          vectorize__binary=[True, False],
    #          vectorize__min_df=MIN_DF,
    #          tfidf=[None, TfidfTransformer()],
    #          reduce_dim=[TruncatedSVD()],
    #          reduce_dim__n_components=N_DIM_OPTIONS
    #          ),
    #     # univariate feature selection
    #     dict(tokens__mapdates=[True, False],
    #          tokens__mapnumbers=[True, False],
    #          tokens__spellcorrect=[False],
    #          tokens__stem=[True, False],
    #          tokens__tokenizer=['word_tokenizer',
    #                             'word_punct_tokenizer'],
    #          vectorize__binary=[True, False],
    #          vectorize__min_df=MIN_DF,
    #          tfidf=[None, TfidfTransformer()],
    #          reduce_dim=[SelectKBest(chi2)],
    #          reduce_dim__k=N_DIM_OPTIONS,
    #          )
    # ]
    # TODO: n_jobs could be an interesting option when on cluster

    # TEST
    param_grid = [
        dict(tokens__mapdates=[False],
             tokens__mapnumbers=[False],
             tokens__spellcorrect=[False],
             tokens__stem=[False],
             tokens__tokenizer=['word_punct_tokenizer'],
             vectorize__binary=[False],
             vectorize__min_df=[1, 2],
             tfidf=[None],
             reduce_dim=[SelectKBest(chi2)],
             reduce_dim__k=[500],
             )
    ]
    grid_search = GridSearchCV(model, cv=CV,
                               param_grid=param_grid,
                               return_train_score=True,
                               scoring=SCORES,
                               refit=False,
                               verbose=verbose)
    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile,
                               subcats=subcats, verbose=verbose)
    grid_search.fit(loader.questions, loader.categoryids)
    return grid_search


if __name__ == "__main__":
    gridsearch = evaluate_model()
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            gridsearch.cv_results_)
