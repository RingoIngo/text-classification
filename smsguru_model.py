"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set.
"""
# Author: Ingo Gühring

from time import gmtime, strftime
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import text_tokenizer_and_cleaner as ttc
import question_loader as ql


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    Taken from scikit learn examples:
    `Feature Union with Heterogeneous Data Sources`

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


class ListVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize a list of features.

    Convert a list of numerical features into the from scikit expected format.
    This list is expected to have a length equal to the number of samples.
    Returns a 2-D array, since format (n,) throws an exception in scikit.
    Type of the return array must be float.
    See also:

    https://stackoverflow.com/questions/22273242/
    scipy-hstack-results-in-typeerror-no-supported-conversion
    -for-types-dtype
    """
    def fit(self, x, y=None):
        return self

    def transform(self, feature_list):
        # this is the from scikit expected format
        feature_vector = np.asarray(feature_list).reshape(len(feature_list), 1)
        return feature_vector.astype(float)

    def get_feature_names(self):
        """Return the name of the feature: `CREATION_HOUR`"""
        return ['CREATION_HOUR']


class QuestionTimeExtractor(BaseEstimator, TransformerMixin):
    """Extract the question & date.

    Takes a list of tuples (question, creation_date) and produces a dict of
    sequences.  Keys are `question` and `time`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, question_dates):
        features = np.recarray(shape=(len(question_dates),),
                               dtype=[('question', object), ('time', object)])
        for i, quest_date_tuple in enumerate(question_dates):
            features['question'][i] = quest_date_tuple['question']
            # only the time is extracted
            features['time'][i] = int(quest_date_tuple['date'].strftime('%H'))

        return features


def _identity(words):
    return words


class SMSGuruModel:
    """Chain processing steps

    Build a `Pipeline` in which all processing steps are chained.
    1. prepare heterogeneous data
        1.1 select textual data (sms questions)
            1.1.1. tokenize and clean text
            1.1.2. vectorize tokens
            1.1.3. tfidf weighting
            1.1.4. TruncatedSVD reduction
        1.2 select creation dates
            1.2.1. vectorize
    2. classifier (optional)

    Parameters
    ----------
    classifier : an classifier, optinal, which is applied as last step
        in the chain. Default is None.
    """

    def __init__(self, classifier=None):
        self.classifier = classifier

    def build(self):
        steps = [
            # Extract the question & its creation time
            ('question_time', QuestionTimeExtractor()),

            # Use FeatureUnion to combine the features from question and time
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for pulling features from the question itself
                    ('question_bow', Pipeline([
                        ('selector', ItemSelector(key='question')),
                        ('tokens', ttc.TextTokenizerAndCleaner()),
                        ('vectorize', CountVectorizer(tokenizer=_identity,
                                                      preprocessor=None,
                                                      lowercase=False)),
                        ('tfidf', TfidfTransformer()),
                        ('reduce_dim', SelectKBest(chi2, k=500)),
                    ])),

                    # Pipeline for creation time
                    ('creation_time', Pipeline([
                        ('selector', ItemSelector(key='time')),
                        ('vectorize', ListVectorizer()),
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights=None,
            )),
        ]

        if self.classifier is not None:
            # Add a classifier to the combined features
            steps.append(('classifier', self.classifier))

        self.model = Pipeline(steps)
        return self


def evaluate_model(qfile='question_train.csv',
                   catfile='category.csv',
                   subcats=True,
                   metadata=True,
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
    model = create_pipeline(classifier=MultinomialNB())
    # param_grid = [
    #     # multivariate feature selection
    #     dict(union__question_bow__tokens__mapdates=[True, False],
    #          union__question_bow__tokens__mapnumbers=[True, False],
    #          union__question_bow__tokens__spellcorrect=[True, False],
    #          union__question_bow__tokens__stem=[True, False],
    #          union__question_bow__tokens__tokenizer=['word_tokenizer',
    #                             'word_punct_tokenizer'],
    #          union__question_bow__vectorize__binary=[True, False],
    #          union__question_bow__vectorize__min_df=MIN_DF,
    #          union__question_bow__tfidf=[None, TfidfTransformer()],
    #          union__question_bow__reduce_dim=[TruncatedSVD()],
    #          union__question_bow__reduce_dim__n_components=N_DIM_OPTIONS
    #          ),
    #     # univariate feature selection
    #     dict(union__question_bow__tokens__mapdates=[True, False],
    #          union__question_bow__tokens__mapnumbers=[True, False],
    #          union__question_bow__tokens__spellcorrect=[False],
    #          union__question_bow__tokens__stem=[True, False],
    #          union__question_bow__tokens__tokenizer=['word_tokenizer',
    #                             'word_punct_tokenizer'],
    #          union__question_bow__vectorize__binary=[True, False],
    #          union__question_bow__vectorize__min_df=MIN_DF,
    #          union__question_bow__tfidf=[None, TfidfTransformer()],
    #          union__question_bow__reduce_dim=[SelectKBest(chi2)],
    #          union__question_bow__reduce_dim__k=N_DIM_OPTIONS,
    #          )
    # ]
    # TODO: n_jobs could be an interesting option when on cluster

    # TEST
    param_grid = [
        dict(union__question_bow__tokens__mapdates=[False],
             union__question_bow__tokens__mapnumbers=[False],
             union__question_bow__tokens__spellcorrect=[False],
             union__question_bow__tokens__stem=[False],
             union__question_bow__tokens__tokenizer=['word_punct_tokenizer'],
             union__question_bow__vectorize__binary=[False],
             union__question_bow__vectorize__min_df=[1, 2],
             union__question_bow__tfidf=[None],
             union__question_bow__reduce_dim=[SelectKBest(chi2)],
             union__question_bow__reduce_dim__k=[500],
             )
    ]
    grid_search = GridSearchCV(model, cv=CV,
                               param_grid=param_grid,
                               return_train_score=True,
                               scoring=SCORES,
                               refit=False,
                               verbose=verbose)
    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile,
                               subcats=subcats,
                               metadata=metadata,
                               verbose=verbose)
    grid_search.fit(loader.questions, loader.categoryids)
    return grid_search


if __name__ == "__main__":
    gridsearch = evaluate_model()
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            gridsearch.cv_results_)
