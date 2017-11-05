"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set.
"""
# Author: Ingo Gühring

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier

import text_tokenizer_and_cleaner as ttc
import question_loader as ql

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
        ('vectorize', CountVectorizer(tokenizer=_identity, preprocessor=None,
                                      lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', SelectKBest(chi2, k=500))
    ]

    if estimator is not None:
        # Add the estimator
        steps.append(('classifier', estimator))
    return Pipeline(steps)


def evaluate_model(qfile, catfile, verbose, subcats):
    CV = 5
    N_DIM_OPTIONS = [500]

    model = create_pipeline(KNeighborsClassifier())
    # TODO:on cluster allow also reduction and spell
    param_grid = [
        # multivariate feature selection
        dict(tokens__mapdates=[True, False],
             tokens__mapnumbers=[True, False],
             tokens__spellcorrector=[False],
             tokens__stemmer=[True, False],
             tokens__tokenizer=['word_tokenizer',
                                'word_punct_tokenizer'],
             vectorize__binary=[True, False],
             vectorize__min_df=[1, 2],
             tfidf=[None, TfidfTransformer()],
             reduce_dim=[TruncatedSVD()],
             reduce_dim__n_components=N_DIM_OPTIONS
             ),
        # univariate feature selection
        dict(tokens__mapdates=[True, False],
             tokens__mapnumbers=[True, False],
             tokens__spellcorrector=[False],
             tokens__stemmer=[True, False],
             tokens__tokenizer=['word_tokenizer',
                                'word_punct_tokenizer'],
             vectorize__binary=[True, False],
             vectorize__min_df=[1, 2],
             tfidf=[None, TfidfTransformer()],
             reduce_dim=[SelectKBest(chi2)],
             reduce_dim__k=N_DIM_OPTIONS,
             ),
    ]
    grid_search = GridSearchCV(model, cv=CV,
                               param_grid=param_grid,
                               return_train_score=True)
    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile,
                               subcats=subcats, verbose=verbose)
    grid_search.fit(loader.questions, loader.categoryids)
    # TODO: save results
    return None


if __name__ == "__main__":
    evaluate_model()
