"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set. The most important class is the SMSGuruModel.
"""
# Author: Ingo Gühring

from time import gmtime, strftime
from itertools import compress
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import text_tokenizer_and_cleaner as ttc
import question_loader as ql


class QuestionCretionDateExtractor(BaseEstimator, TransformerMixin):
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

        # this is the from scikit expected format
        time_vector = np.asarray(time_list).reshape(len(time_list), 1)
        return time_vector

    def get_feature_names(self):
        """Return the name of the feature: `CREATION_HOUR`"""
        return ['CREATION_HOUR']


def _identity(words):
    return words


# for multiclass it holds:
# recall_micro = precision_micro = f1_micro = accuracy
SCORES = ['recall_macro', 'precision_macro', 'f1_macro', 'f1_micro']


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
            1.1.4. TruncatedSVD reduction
        1.2 select creation dates
            1.2.1. vectorize
    2. classifier (optional)

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

    def __init__(self, classifier=MultinomialNB(), metadata=True):
        self.classifier = classifier
        self.metadata = metadata
        self.model = self._build(self.classifier, self.metadata)
        self.is_fitted = False

    def _build(self, classifier, metadata):
        """build the model"""
        steps = [
            # Extract the question & its creation time
            ('question_created_at', QuestionCretionDateExtractor()),

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
                        ('selector', ItemSelector(key='created_at')),
                        ('vectorize', CreationDateVectorizer()),
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights=None,
            )),
        ]

        if classifier is not None:
            # Add a classifier to the combined features
            steps.append(('classifier', classifier))

        pipeline = Pipeline(steps)

        if not metadata:
            pipeline.set_params(union__creation_time=None)

        return Pipeline(steps)

    def get_feature_names(self):
        """Return the feature names if interpretable

        In case none or the univariate feature selection method is used
        the names of the words whose counts (or transformed counts,
        e.g.tfidf) represent the features are returned.
        If `metadate` is True the `get_feature_mames()` method of the class
        CreationDateVectorizer is used and its output appended to the list
        of feature names.'

        Returns
        ---------
        featurenames : A numpy array containing the feature names as strings
        """
        if not self.is_fitted:
            print("invoke fit_transform first!")
            return

        if (isinstance(self.model.named_steps['union'].
                       transformer_list[0][1].named_steps['reduce_dim'],
                       TruncatedSVD)):
            # no interpretable features names
            featurenames = None
        else:
            featurenames = (self.model.named_steps['union'].
                            transformer_list[0][1].named_steps['vectorize'].
                            get_feature_names())

        if (isinstance(self.model.named_steps['union'].
                       transformer_list[0][1].named_steps['reduce_dim'],
                       SelectKBest)):
            feature_mask = (self.model.named_steps['union'].
                            transformer_list[0][1].named_steps['reduce_dim'].
                            get_support())
            featurenames = list(compress(featurenames, feature_mask))

        if ((self.model.named_steps['union'].transformer_list[1][1]
                is not None) and featurenames is not None):
            metadata_label = (self.model.named_steps['union'].
                              transformer_list[1][1].named_steps['vectorize'].
                              get_feature_names())
            featurenames = featurenames + metadata_label

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

    def set_question_loader(self, qfile='question_train.csv',
                            catfile='category.csv',
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

    def gridsearch(self, param_grid, verbose=10, CV=5, n_jobs=1):
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
        # TODO: n_jobs could be an interesting option when on cluster
        self.n_jobs_ = n_jobs
        self.param_grid_ = param_grid

        self.grid_search_ = GridSearchCV(self.model, cv=self.CV_,
                                         param_grid=self.param_grid_,
                                         return_train_score=True,
                                         scoring=SCORES,
                                         refit=False,
                                         n_jobs=self.n_jobs_,
                                         verbose=verbose)

        self.grid_search_.fit(self.question_loader_.questions,
                              self.question_loader_.categoryids)
        return self


if __name__ == "__main__":
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

    grid = [{**base_grid, **univariate}, {**base_grid, **multivariate}]

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

    sms_guru_model = (SMSGuruModel(classifier=classifier, metadata=metadata).
                      set_question_loader(subcats=subcats).
                      gridsearch(grid))
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    np.save('./results/gridsearch/' + current_time + 'grids_cv.npy',
            sms_guru_model.grid_search_.cv_results_)
    with open("./results/gridsearch/gridsearches.txt", "a") as report:
        report.write("""performed at: {}, non_grid_params:  subcats: {},
                      metadata: {}, classifier: MultinomialNB\n"""
                     .format(current_time, subcats, metadata))
