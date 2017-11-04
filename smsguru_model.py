"""
The :mod: `smsguru_model` implements and evaluates the model for the
sms guru data set.
"""
# Author: Ingo Gühring

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import text_tokenizer_and_cleaner as ttc


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
        ('reduction', TruncatedSVD(n_components=10000))
    ]

    if estimator is not None:
        # Add the estimator
        steps.append(('classifier', estimator))
    return Pipeline(steps)


def evaluate_model(estimator):
    model = create_pipeline()
    # TODO:on cluster allow also reduction and spell
    param_grid = dict(tokens__mapdates=[True, False],
                      tokens__mapnumbers=[True, False],
                      tokens__spellcorrector=[False],
                      tokens__stemmer=[True, False],
                      tokens__tokenizer=['word_tokenizer',
                                         'word_punct_tokenizer'],
                      vectorize__binary=[True, False],
                      tfidf=[None, TfidfTransformer()],
                      reduction=[None])
    grid_search = GridSearchCV(model, param_grid=param_grid)
    return None


if __name__ == "__main__":
    evaluate_model()
