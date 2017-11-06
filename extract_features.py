"""
The :mod:`extract_features` module implements the function
`extract_features`
"""
# Author: Ingo Gühring
import numpy as np
import plac

import smsguru_model
import question_loader as ql


@plac.annotations(
    binary=(None, 'flag', 'bi'),
    mapdates=(None, 'option', 'md'),
    mapnumbers=(None, 'option', 'mn'),
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
                     mapdates=True,
                     mapnumbers=False,
                     metadata=True,
                     spellcorrector=False,
                     stemmer=True,
                     subcats=True,
                     svd=False,
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
    if not tfidf:
        model.set_params(tfidf=None)
    # TODO: integrade reduce_dim
    model.set_params(reduce_dim=None)
    # get features
    features = model.fit_transform(loader.questions, loader.categoryids)
    # get feature names
    if svd:
        featurenames = None
    else:
        featurenames = model.named_steps['vectorize'].get_feature_names()
    if verbose:
        print("feature matrix size {}".format(features.T.shape))
        print("featurenames size {}".format(len(featurenames)))
        print("categoryids size {}".format(len(loader.categoryids)))
        print("categories size: {}".format(len(loader.categories)))
        print("number of questions: {}".format(len(loader.questions)))
        print("filtered because of min_df = {}:".format(min_df))
        print(model.named_steps['vectorize'].stop_words_)
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
