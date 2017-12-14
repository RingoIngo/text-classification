"""
The :mod:`extract_features` module implements the function
`extract_features`
"""
# Author: Ingo Gühring
import argparse

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

from smsguru_model import SMSGuruModel


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     binary=False,
                     dim=500,
                     mapdates=True,
                     mapnumbers=False,
                     metadata=False,
                     reduce_dim='chi2',
                     spellcorrect=False,
                     stem=True,
                     subcats=True,
                     tfidf=False,
                     min_df=2,
                     tokenizer='word_punct_tokenizer',
                     outfile='features.npz',
                     verbose=False):
    """Extract features from files with questions and categories

    Extract features from data in SMSGuru format and save the computed
    features, featurenames, categoryids and categories.

    Parameters
    -----------

    qfile : csv file containing the questions and the
       corresponding categories. By default the file
       'question_train.csv' is loaded.

    qcatfile : required, but unused.

    catfile : csv file containing the categoires of the questions.
        a category has a category_nam (string), a category_id (int)
        and a parent_id (int). if the parent_id eqauals zero the
        category has no parent.

    binary : boolean, if True a `set of words` model is used
        instead of a bag of words model for vectorization of
        the textual features. Default is False.

    dim : integer, which sets the number of dimensions to which
        the feature space is reduced in the dimensionality
        reduction step. Defalt is 500.

    mapdate : boolean, if True, all tokens whose format is in a
        specific set of date formats are mapped to the dummy
        date `00.00.00`. Default is True.

    mapnumbers : boolean, if True, all tokens containing a number
        while not being in a date format are mapped to the dummy
        number `0`. Default is False.

    metadata : boolean, if True also the creation hour of the
        samples is extracted from the csv files and used as
        a feature.

    reduce_dim : string in ['chi2', 'trunSVD', 'None'].
        Specifies the method used to reduce the dimensionality
        of the feature space. Default is 'chi2'.

    spellcorrect : boolean, if True, tokens are spell corrected.
        Default is False.

    stem : boolean, if True tokens are stemmed. Default is True.

    subcats : boolean, if True, the subcategories are used as labels
        for the samples. If False, the parent categories are used.
        Default is True.

    tfidf : boolean, if True, tf-idf weighting is applied to
        numerical feature vectore.

    min_df : integer. When building the vocabulary only tokens
        with a document frequency greater or equal then min_df
        are taken into account.

    tokenizer : string in ['word_punct_tokenizer', 'word_tokenizer'].
        Specifies which tokenizer from the nltk package is used.

    outfile : string, name of the file the output is saved to.
        should end with 'npz'. Default is 'features.npz'.

    verbose : boolean, if True output about the state of the program
        and the extracted features is printed to the console.
        Default is False.

    Returns
    ---------
    NOTHING, BUT SAVES TO outfile:

    features : d x n matrix (numpy array) that contains the feature
        vectors, where d is the number of features and n is the
        number of data points.

    featurenames : list, containing strings with d entries that
        contains for each feature dimension a human-readable
        description.

    categoryids : 1 x n numpy array, that contains for each feature vector
        its category id.

    categories : dictionary, where the keys are the entries in
        `categoryids` and the values are a textual description
        of each category.
    """

    sms_guru_model = SMSGuruModel(classifier=None, metadata=metadata)
    sms_guru_model.set_question_loader(qfile=qfile, catfile=catfile,
                                       subcats=subcats, verbose=verbose)
    # tokens is the name of the first transformation in the pipeline
    sms_guru_model.model.set_params(
        union__bow__tokens__mapdates=mapdates,
        union__bow__tokens__mapnumbers=mapnumbers,
        union__bow__tokens__spellcorrect=spellcorrect,
        union__bow__tokens__stem=stem,
        union__bow__tokens__tokenizer=tokenizer,
        union__bow__vectorize__binary=binary,
        union__bow__vectorize__min_df=min_df,
    )
    # term frequency weighting
    if not tfidf:
        sms_guru_model.model.set_params(union__bow__tfidf=None)

    # dimension reduction
    if reduce_dim == 'None':
        sms_guru_model.model.set_params(reduce_dim=None)
    elif reduce_dim == 'trunSVD':
        sms_guru_model.model.set_params(
            reduce_dim=TruncatedSVD(n_components=dim))
    elif reduce_dim == 'chi2':
        sms_guru_model.model.set_params(
            reduce_dim=SelectKBest(chi2, k=dim))

    features = sms_guru_model.fit_transform()
    featurenames = sms_guru_model.get_feature_names()

    if verbose:
        print("feature matrix size {}".format(features.T.shape))
        if featurenames is not None:
            print("featurenames size {}".format(len(featurenames)))
        else:
            # e.g. PCA
            print("no interpretable featurenames available")
        print("categoryids size {}".format(
            len(sms_guru_model.question_loader_.categoryids)))
        print("categories size: {}".format(
            len(sms_guru_model.question_loader_.categories)))
        print("number of questions: {}".format(
            len(sms_guru_model.question_loader_.questions)))
        print("filtered because of min_df = {}:".format(min_df))
        print(sms_guru_model.get_filtered_words())
        if featurenames is not None:
            print("feature names: {}".format(featurenames))
    # save extracted features
    np.savez(outfile, features=features.T.toarray(),
             featurenames=featurenames,
             categoryids=sms_guru_model.question_loader_.categoryids[None, :],
             categories=sms_guru_model.question_loader_.categories)


# run extract_features method if module is executed as a script
# if non-default values are wanted, use command line options
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('-d', '--dim', type=int, default=argparse.SUPPRESS)

    parser.add_argument('-md', '--mapdates', action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('-mn', '--mapnumbers', action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('-m', '--metadata', action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('-r', '--reduce-dim',
                        choices=['chi2', 'trunSVD', 'None'],
                        default=argparse.SUPPRESS)

    parser.add_argument('-s', '--spellcorrect', action='store_true',
                        default=argparse.SUPPRESS)

    # this is not really necessary, since stemming is used by default
    parser.add_argument('--stem', action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('--no-stem', action='store_false', dest='stem',
                        default=argparse.SUPPRESS)

    parser.add_argument('--parent-cats', dest='subcats', action='store_false',
                        default=argparse.SUPPRESS)

    parser.add_argument('--tfidf', action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('--min-df', type=int, default=argparse.SUPPRESS)

    parser.add_argument('-t', '--tokenizer',
                        choices=['word_punct_tokenizer', 'word_tokenizer'],
                        default=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', action='store_true',
                        default=argparse.SUPPRESS)

    try:
        extract_features(**vars(parser.parse_args()))
    except ValueError:
        print("""Invalid option combination! Note that univariate feature
              reduction uses the chi2 method which expects non-negative
              features. Since the creation hour which is used in metadata is
              modeled as a 2dim cyclic feature it also contains negative
              values. So the metadata option can not be used together with
              univariate feature selection.""")
