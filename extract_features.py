"""
The :mod:`extract_features` module implements the function
`extract_features`
"""
# Author: Ingo Gühring
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import plac
import pprint

import smsguru_model
import question_loader as ql


@plac.annotations(
    binary=(None, 'flag', 'bi'),
    dim=(None, 'option', None, int),
    mapdates=(None, 'option', 'md'),
    mapnumbers=(None, 'option', 'mn'),
    reduce_dim=(None, 'option', 're_d', str, ['chi2', 'trunSVD', 'None']),
    spellcorrector=(None, 'flag', 'sp'),
    stemmer=(None, 'option', None, bool),
    subcats=(None, 'option', None, bool),
    tfidf=(None, 'flag'),
    min_df=(None, 'option', 'min_df', int),
    tokenizer=(None, 'option', 'tok', str, ['word_punct_tokenizer',
                                            'word_tokenizer']),
    verbose=('', 'flag', 'v'))
def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     binary=False,
                     dim=500,
                     mapdates=True,
                     mapnumbers=False,
                     metadata=True,
                     reduce_dim='chi2',
                     spellcorrector=False,
                     stemmer=True,
                     subcats=True,
                     tfidf=False,
                     min_df=1,
                     tokenizer='word_punct_tokenizer',
                     outfile='features.npz',
                     verbose=False):
    """Extract features from files with questions and categories
    TODO: add doc when function finished
    """
    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile,
                               subcats=subcats, verbose=verbose)
    model = smsguru_model.create_pipeline()
    # tokens is the name of the first transformation in the pipeline
    model.set_params(tokens__mapdates=mapdates,
                     tokens__mapnumbers=mapnumbers,
                     tokens__spellcorrector=spellcorrector,
                     tokens__stemmer=stemmer,
                     tokens__tokenizer=tokenizer,
                     vectorize__binary=binary,
                     vectorize__min_df=min_df)
    # term frequency weighting
    if not tfidf:
        model.set_params(tfidf=None)

    # dimension reduction
    if reduce_dim == 'None':
        model.set_params(reduce_dim=None)
    elif reduce_dim == 'trunSVD':
        model.set_params(reduce_dim=TruncatedSVD(n_components=dim))
    elif reduce_dim == 'chi2':
        model.set_params(reduce_dim=SelectKBest(chi2, k=dim))

    # get features
    features = model.fit_transform(loader.questions, loader.categoryids)
    # get feature names
    if reduce_dim == 'None':
        featurenames = model.named_steps['vectorize'].get_feature_names()
    elif reduce_dim == 'trunSVD':
        # no interpretable feature names
        featurenames = None
    elif reduce_dim == 'chi2':
        featurenames = np.asarray(
            model.named_steps['vectorize'].get_feature_names())
        featurenames = featurenames[
            model.named_steps['reduce_dim'].get_support()]

    if verbose:
        print("feature matrix size {}".format(features.T.shape))
        print("featurenames size {}".format(len(featurenames)))
        print("categoryids size {}".format(len(loader.categoryids)))
        print("categories size: {}".format(len(loader.categories)))
        print("number of questions: {}".format(len(loader.questions)))
        print("filtered because of min_df = {}:".format(min_df))
        print(model.named_steps['vectorize'].stop_words_)
        print("feature names: {}".format(featurenames))
        pprint.pprint(featurenames)
    # save extracted features
    np.savez(outfile, features=features.T.toarray(),
             featurenames=featurenames,
             categoryids=loader.categoryids[None, :],
             categories=loader.categories)


# run extract_features method if module is executed as a script
# put non-default input here in function
if __name__ == "__main__":
    # extract_features(tokenizer='word_tokenizer',
    #                  tfidf=True, mapnumbers=True, min_df=2)
    plac.call(extract_features)
