"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set. The most important class is the SMSGuruModel.
"""
# Author: Ingo Guehring

from time import gmtime, strftime
from itertools import compress
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer

import model.text.text_tokenizer_and_cleaner as ttc
import model.question_loader as ql


class QuestionCreationDateExtractor(BaseEstimator, TransformerMixin):
    """Extract the question & date.

    Takes a list of tuples (question, creation_date) and produces a dict of
    sequences.  Keys are `question` and `time`.

    The reason for the class in the context of the model:
        The data is given to the model as a list of tuples,
        where each entry in a tuple is a feature. Those feature
        may then be expanded to multiple features (bow) or directly
        used (creation time). To be able to process different features
        differently the transform method of this class produces a
        dictionary of sequences, where the keys are the different
        feature names. When the processing is done the ItemSelector class
        selects the features that are needed for a specific processing step
        by key from the output the transform method of this class.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, question_dates):
        """Transform from grouped by sample to grouped by feature

        Parameters
        -----------
        question_dates : a list of tuples, where each tuple corresponds
            to a sample and each entry of a tuple to a feature.

        Returns
        -----------
        features : a dictionary of sequences where the keys are the
            feature names and the sequences are the features.
        """
        features = np.recarray(shape=(len(question_dates),),
                               dtype=[('question', object),
                                      ('created_at', object)])
        for i, quest_date_tuple in enumerate(question_dates):
            features['question'][i] = quest_date_tuple['question']
            features['created_at'][i] = quest_date_tuple['created_at']

        return features


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


class CreationDateVectorizer(BaseEstimator, TransformerMixin):
    """Transform and vectorize a list of creation dates.

    Convert a list of creation dates into the from scikit expected format.
    This list is expected to have a length equal to the number of samples.
    Returns a 2-D array, since format (n,) throws an exception in scikit.
    type of the return array must be float.
    see also:

    https://stackoverflow.com/questions/22273242/
    scipy-hstack-results-in-typeerror-no-supported-conversion
    -for-types-dtype

    Before the vectorization a transforamtion is applied to the dates.
    Only the hour is extracted and used as a feature.

    Note that if other meta data is to be used the processing of the features
    should be implemented in this class, i.e.
    CreationDateVectorizer --> QuestionStatsVectorizer
    """

    def fit(self, x, y=None):
        return self

    def transform(self, date_list):
        """"Transform the date list into an array containing the hours.

        Parameters
        ------------
        date_list : list containing the creation dates. Length is expected
            to be the sample size.

        Returns
        -----------
        time_vector : a numpy 2D array (n,1) containing the hours of the
            creation dates as floats. n is the number of samples.
        """
        # extract creation hour
        try:
            time_list = [float(date.strftime('%H')) for date in date_list]
        except AttributeError:
            print("""meta data has not been extracted from the data source!
                  If this feature is used in the model, the QuestionLoader
                  should get metadate=True""")

        time_vector = np.asarray(time_list).reshape(len(time_list), 1)

        # apply a cyclic transformation so that 0 and 23 are close
        # see https://ianlondon.github.io/blog/
        # encoding-cyclical-features-24hour-time/
        cyclic_feature = np.concatenate((np.sin(2*np.pi*time_vector/24),
                                         np.cos(2*np.pi*time_vector/24)),
                                        axis=1)
        return cyclic_feature

    def get_feature_names(self):
        """Return the name of the cyclic feature components"""
        return ['CREATION_HOUR_SIN', 'CREATION_HOUR_COS']


class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


def _identity(words):
    return words


class NamedPipeline(Pipeline):
    """Extend `Pipleline` class with a `get_feature_names` method

    """

    def get_feature_names(self):
        """Return the feature names

        Traverse the chained transformers from the end and check
        `get_feature_names` method is implemented. If so, return
        feature names.

        Returns
        ---------
        featurenames : a list, containing the feature names as strings,
                or None, if no transformer implements the
                `get_feature_names` method
        """
        for name, transformer in reversed(self.steps):
            if (transformer is not None and
                    hasattr(transformer, 'get_feature_names')):
                return transformer.get_feature_names()
        return None


def nested_cv(estimator, param_grid, cv, X, y, scoring):
    """
    Perform a nested gridsearch to evaluate an estimator with param_grid

    In contrast to a non-nested gridsearch this method does NOT find
    the best parameter combination from the grid, but gives an estimate
    of the generalization error of an estimator with associated grid.
    The output can be used to compare different estimators
    (e.g. SVM vs. kNN) without taking a specific optimized hyper paramter
    combination into account, but the whole range of possible parameters
    specified in the grid. See e.g.
    `http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_nested_cross_validation_iris.html`

    Parameters
    ----------
    estimator : estimator which is evaluated

    param_grid : dictionary of parameters of which the best one from the
        the inner cv is used to give a score in the outer cv.
        See the sklearn Pipeline documentation on how
        parameters need to be specified.

    cv : number of folds for crossvalitdation in inner and in outer cv

    X : feature matrix

    y : labels

    scoring: scoring function

    Returns
    ---------
    nested_scores : an array of shape=(cv,) which contains the scores
        of the validations in the outer cv
    """
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                       n_jobs=-1, scoring=scoring, verbose=100)
    return cross_val_score(clf, X=X, y=y, cv=cv, scoring=scoring, verbose=100)


PARENT_CLASSES = [13, 12, 11, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 74]
PARENT_LABEL_BINARIZER = LabelBinarizer().fit(PARENT_CLASSES)


# for multiclass it holds:
# recall_micro = precision_micro = f1_micro = accuracy
# SCORES = ['recall_macro', 'precision_macro', 'f1_macro', 'f1_micro']
def roc_auc(y_true, y_score):
    """Wrapper function for roc_auc_score with average=None"""
    y_true_bin = PARENT_LABEL_BINARIZER.transform(y_true)
    y_score_bin = PARENT_LABEL_BINARIZER.transform(y_score)
    return roc_auc_score(y_true_bin, y_score_bin, average=None)


def roc_auc_micro(estimator, X, y):
    """Wrapper function for roc_auc_score with average='micro'"""
    y_true_bin = PARENT_LABEL_BINARIZER.transform(y)
    y_score_bin = estimator.predict_proba(X)
    return roc_auc_score(y_true_bin, y_score_bin, average='micro')


# def roc_auc_macro(y_true, y_score):
#     """Wrapper function for roc_auc_score with average='macro'"""
#     y_true_bin = PARENT_LABEL_BINARIZER.transform(y_true)
#     y_score_bin = PARENT_LABEL_BINARIZER.transform(y_score)
#     return roc_auc_score(y_true_bin, y_score_bin, average='macro')
#
#
def roc_auc_macro(estimator, X, y):
    """Wrapper function for roc_auc_score with average='macro'"""
    y_true_bin = PARENT_LABEL_BINARIZER.transform(y)
    y_score_bin = estimator.predict_proba(X)
    return roc_auc_score(y_true_bin, y_score_bin, average='macro')


# SCORES = {'recall_macro': 'recall_macro',
#           'precision_macro': 'precision_macro',
#           'f1_macro': 'f1_macro',
#           'f1_micro': 'f1_micro',
#           # 'roc_auc_micro': make_scorer(roc_auc_micro),
#           # 'roc_auc': make_scorer(roc_auc)
#           }


# SCORES_BIN = {'recall_macro': 'recall_macro',
#               'precision_macro': 'precision_macro',
#               'f1_macro': 'f1_macro',
#               'f1_micro': 'f1_micro',
#               'roc_auc_micro': make_scorer(roc_auc_micro),
#               'roc_auc_macro': make_scorer(roc_auc_macro),
#               # 'roc_auc': make_scorer(roc_auc)
#               }


SCORES_BIN = {'roc_auc_micro': roc_auc_micro,
              'roc_auc_macro': roc_auc_macro,
              }


class SMSGuruModel:
    """Contains all the steps from file reading to the model

    Reading in the question and categories and building and
    evaluating a model is handled by this class. The model is
    as follows:

    Build a `Pipeline` in which all processing steps are chained.
    1. prepare heterogeneous data
        1.1 select textual data (sms questions)
            1.1.1. tokenize and clean text
            1.1.2. vectorize tokens
            1.1.3. tfidf weighting
        1.2 select creation dates
            1.2.1. vectorize
    2. dimension reduction
    3. classifier (optional)

    See the __main__ for an example how the class can be used.

    Parameters
    ----------
    classifier : an classifier, optinal, which is applied as last step
        in the chain. Default is None.

    metadata : a boolean, optional, which determines if beside the question
        also metadata is used. In this case the metadata consists simply
        of the creation date of a question from which only the hour is used.
        More metadata could possibly be used, e.g. the length of the question.
        Unlike many other parameters that change the behavior of the model,
        this parameter has to be set when instantiating a SMSGuruModel,
        because it influences also the data extraction from the files.
        If set True only samples with a valid creation date can be used.
        This reduces the number of usable samples (about 36 less).

    Attributes
    ----------
    model : Pipeline, chains the transformation and classification
        operations

    question_loader_ : QuestionLoader, contains the data read from
        input files and the datafile related logic. See module
        `question_loader.py` for more information. The model can
        only be fitted if `question_loader_ is set. See
        `set_question_loader()` class method.

    is_fitted : boolean, True if the attribure `model` has been fitted
        and transformed. See `fit_transform()` class method.

    CV_ : int, number of folds used by the crossvalidation in GridsearchCV
        in class mmethod `gridsearch()`

    n_jobs_ : int, number of jobs used by GridsearchCV in class method
        `gridsearch()`

    param_grid_ : dictionary, used by GridsearchCV in class method
        `gridsearch()`

   grid_search_ : GridsearchCV, used for a gridsearch in class mehtod
        `gridsearch()`

    """

    def __init__(self, classifier=MultinomialNB(),
                 pre_reduction=None,
                 reduction=SelectKBest(chi2, k=500),
                 metadata=True,
                 memory=False,
                 to_dense=False,
                 binarize=False):
        self.classifier = classifier
        self.pre_reduction = pre_reduction
        self.reduction = reduction
        self.metadata = metadata
        self.memory = memory
        self.to_dense = to_dense
        self.binarize = binarize
        self.model = self._build(self.classifier,
                                 self.pre_reduction,
                                 self.reduction,
                                 self.metadata,
                                 self.memory,
                                 to_dense)
        self.is_fitted = False
        self.CV_ = None
        self.n_jobs_ = None
        self.grid_search_ = None
        self.param_grid_ = None

    def _build(self, classifier, pre_reduction, reduction, metadata, memory,
               to_dense):
        """build the model"""
        steps = [
            # Extract the question & its creation time
            ('question_created_at', QuestionCreationDateExtractor()),

            # Use FeatureUnion to combine the features from question and time
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for pulling features from the question itself
                    ('bow', NamedPipeline([
                        ('selector', ItemSelector(key='question')),
                        ('tokens', ttc.TextTokenizerAndCleaner()),
                        ('vectorize', CountVectorizer(tokenizer=_identity,
                                                      min_df=2,
                                                      preprocessor=None,
                                                      lowercase=False)),
                        ('tfidf', None),
                    ])),

                    # Pipeline for creation time
                    ('creation_time', NamedPipeline([
                        ('selector', ItemSelector(key='created_at')),
                        ('vectorize', CreationDateVectorizer()),
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights=None,
            )),
            ('pre_reduce_dim', pre_reduction),
            ('to_dense', DenseTransformer()),
            ('reduce_dim', reduction),
        ]

        if classifier is not None:
            # Add a classifier to the combined features
            steps.append(('classifier', classifier))

        pipeline = Pipeline(steps, memory)

        if not metadata:
            pipeline.set_params(union__creation_time=None)

        if not to_dense:
            pipeline.set_params(to_dense=None)

        return Pipeline(steps)

    def get_feature_names(self):
        """Return the feature names if interpretable

        In case none or the univariate feature selection method is used
        the names of the selected features prefixed by the feature type
        (i.e. bow or creation_time) are returned. The feature names of the
        features in bow are the words whose counts (or transformed counts,
        e.g.tfidf) represent the features.

        Returns
        ---------
        featurenames : A numpy array containing the feature names as strings
        """
        if not self.is_fitted:
            print("invoke fit_transform first!")
            return

        if isinstance(self.model.named_steps['reduce_dim'], TruncatedSVD):
            # no interpretable features names
            return None
        else:
            featurenames = self.model.named_steps['union'].get_feature_names()

        if isinstance(self.model.named_steps['reduce_dim'], SelectKBest):
            feature_mask = (self.model.named_steps['reduce_dim'].get_support())
            featurenames = list(compress(featurenames, feature_mask))

        return np.asarray(featurenames)

    def get_filtered_words(self):
        """Return by min document frequency filtered words.

        Returns
        ----------
        filtered_words: list, words that were filtered because the df attribut
            of the vectorizer in bow was set higher than one.
        """
        filtered_words = (self.model.named_steps['union'].
                          transformer_list[0][1].named_steps['vectorize'].
                          stop_words_)
        return filtered_words

    def set_question_loader(self, qfile='./data/question_train.csv',
                            catfile='./data/category.csv',
                            subcats=True,
                            verbose=False,
                            ):
        """
        Set the `question_loader_` attribute.

        Read in a dataset from files and store it in
        QuestionLoader class.

        Parameters
        ----------
        qfile : csv file containing the questions, their creation
            date and their main category

        catfile : csv file containing the categories and their
            parent realtions

        subcats : boolean, determines weather main or subcategories
            are stored

        verbose : boolean, determines if information is printed during
            the reading of the files

        Returns
        ---------
        self
        """
        self.question_loader_ = ql.QuestionLoader(qfile=qfile,
                                                  catfile=catfile,
                                                  subcats=subcats,
                                                  binarize=self.binarize,
                                                  metadata=self.metadata,
                                                  verbose=verbose)
        return self

    def fit_transform(self):
        """
        Learn the vocabulary dictionary, reduce feature dimension and
        return feature matrix

        Returns
        ---------
        X : array, [n_samples, n_features]
            Feature matrix
        """
        if not hasattr(self, 'question_loader_'):
            print("Set question loader first!")
            return

        if (not self.question_loader_.metadata and
                (self.model.named_steps['union'].transformer_list[1][1])
                is not None):
            print("""When creation time is used in the model the meta
                  data option in question loader must also be True""")
            return

        self.is_fitted = True
        return self.model.fit_transform(self.question_loader_.questions,
                                        self.question_loader_.categoryids)

    def gridsearch(self, param_grid, verbose=100, CV=5, n_jobs=1):
        """
        Perform a gridsearch

        Parameters
        ----------
        param_grid : dictionary of parameters tested in GridsearchCV on
            `model`. See the sklearn Pipeline documentation on how
            parameters need to be specified.

        verbose : int, verbosity

        CV : number of folds for crossvalitdation in GridsearchCV

        n_jobs : number of jobs used in GridsearchCV

        Returns
        ---------
        self
        """
        if not hasattr(self, 'question_loader_'):
            print("Set question loader first!")
            return

        if self.is_fitted:
            print("model has already been fitted")

        self.CV_ = CV
        self.n_jobs_ = n_jobs
        self.param_grid_ = param_grid
        # if labels are binary we can use the AUC metrics
        # if self.binarize:
        #     scores = SCORES_BIN
        # else:
        #     scores = SCORES

        self.grid_search_ = GridSearchCV(self.model, cv=self.CV_,
                                         param_grid=self.param_grid_,
                                         return_train_score=True,
                                         scoring=SCORES_BIN,
                                         refit=False,
                                         error_score=-1,
                                         n_jobs=self.n_jobs_,
                                         verbose=verbose)

        self.grid_search_.fit(self.question_loader_.questions,
                              self.question_loader_.categoryids)
        return self

    def nested_cv(self, param_grid, CV=5, scoring='f1_macro'):
        """
        Perform a nested gridsearch to evaluate an estimator with param grid

        In contrast to a non-nested gridsearch this method does NOT find
        the best parameter combination from the grid, but gives an estimate
        of the generalization error of an estimator with associated grid.
        The output can be used to compare different estimators
        (e.g. SVM vs. kNN) without taking a specific optimized hyper paramter
        combination into account, but the whole range of possible parameters
        specified in the grid. See e.g.
        `http://scikit-learn.org/stable/auto_examples/model_selection/
        plot_nested_cross_validation_iris.html`

        Parameters
        ----------
        param_grid : dictionary of parameters of which the best one from the
            the inner cv is used to give a score in the outer cv.
            See the sklearn Pipeline documentation on how
            parameters need to be specified.

        cv : number of folds for crossvalitdation in inner and outer cv


        Returns
        ---------
        nested_scores : an array of shape=(CV,) which contains the scores
            of the validations in the outer cv
        """
        if not hasattr(self, 'question_loader_'):
            print("Set question loader first!")
            return

        return nested_cv(estimator=self.model,
                         param_grid=param_grid,
                         cv=CV,
                         X=self.question_loader_.questions,
                         y=self.question_loader_.categoryids,
                         scoring=scoring)


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
    N_DIM_OPTIONS = [10, 50, 100, 200, 500, 1000, 1500, 2500, 5000]

    # this choice is based on [Seb99]
    MIN_DF = [1, 2, 3]

    base_grid = (
        dict(union__bow__tokens__mapdates=[True, False],
             union__bow__tokens__mapnumbers=[True, False],
             union__bow__tokens__spellcorrect=[True, False],
             union__bow__tokens__stem=[True, False],
             # union__bow__tokens__tokenizer=['word_tokenizer',
             #                                         'word_punct_tokenizer'],
             union__bow__tokens__tokenizer=['word_punct_tokenizer'],
             # union__bow__vectorize__binary=[True, False],
             union__bow__vectorize__min_df=MIN_DF,
             union__bow__tfidf=[None, TfidfTransformer()],
             )
    )

    univariate = (
        dict(union__bow__reduce_dim=[SelectKBest(chi2)],
             union__bow__reduce_dim__k=N_DIM_OPTIONS)
    )

    multivariate = (
        dict(union__bow__reduce_dim=[TruncatedSVD()],
             union__bow__reduce_dim__n_components=N_DIM_OPTIONS)
    )

    grid = [merge_two_dicts(base_grid, univariate),
            merge_two_dicts(base_grid, multivariate)]

    # TEST
    test_param_grid = [
        dict(union__bow__tokens__mapdates=[False],
             union__bow__tokens__mapnumbers=[False],
             union__bow__tokens__spellcorrect=[False],
             union__bow__tokens__stem=[False],
             union__bow__tokens__tokenizer=['word_punct_tokenizer'],
             union__bow__vectorize__binary=[False],
             union__bow__vectorize__min_df=[1, 2],
             union__bow__tfidf=[None],
             union__bow__reduce_dim=[SelectKBest(chi2)],
             union__bow__reduce_dim__k=[30],
             ),
    ]

    sms_guru_model = (SMSGuruModel(classifier=classifier, metadata=metadata).
                      set_question_loader(subcats=subcats).
                      gridsearch(param_grid=grid, n_jobs=-1))
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            sms_guru_model.grid_search_.cv_results_)
    with open("./results/gridsearch/gridsearches.txt", "a") as report:
        report.write("""performed at: {}, non_grid_params:  subcats: {},
                      metadata: {}, classifier: MultinomialNB\n"""
                     .format(current_time, subcats, metadata))
