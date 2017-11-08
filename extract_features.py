"""
The :mod:`extract_features` module implements the function
`extract_features`
"""
# Author: Ingo Gühring
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import plac
import pdb

from smsguru_model import SMSGuruModel


@plac.annotations(
    binary=(None, 'flag', 'bi'),
    dim=(None, 'option', None, int),
    # TODO: this doest work in terminal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mapdates=(None, 'option', 'md'),
    mapnumbers=(None, 'option', 'mn'),
    reduce_dim=(None, 'option', 're_d', str, ['chi2', 'trunSVD', 'None']),
    spellcorrect=(None, 'flag', 'sp'),
    stem=plac.Annotation(None, 'option', None, str, ['True', 'False']),
    subcats=plac.Annotation(None, 'option', None, str, ['True', 'False']),
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
                     spellcorrect=False,
                     stem='True',
                     subcats='True',
                     tfidf=False,
                     min_df=1,
                     tokenizer='word_punct_tokenizer',
                     outfile='features.npz',
                     verbose=False):
    """Extract features from files with questions and categories
    """
    # TODO: add doc when function finished
    # this cumbersome construction is due to plac annotations
    stem = True if stem == 'True' else False
    subcats = True if subcats == 'True' else False
    sms_guru_model = SMSGuruModel(classifier=None).set_question_loader(
                                                        qfile=qfile,
                                                        catfile=catfile,
                                                        metadata=metadata,
                                                        subcats=subcats,
                                                        verbose=verbose,
                                                        )
    # tokens is the name of the first transformation in the pipeline
    sms_guru_model.model.set_params(
        union__question_bow__tokens__mapdates=mapdates,
        union__question_bow__tokens__mapnumbers=mapnumbers,
        union__question_bow__tokens__spellcorrect=spellcorrect,
        union__question_bow__tokens__stem=stem,
        union__question_bow__tokens__tokenizer=tokenizer,
        union__question_bow__vectorize__binary=binary,
        union__question_bow__vectorize__min_df=min_df,
    )
    # metadata
    if not metadata:
        sms_guru_model.model.set_params(union__creation_time=None)
    # term frequency weighting
    if not tfidf:
        sms_guru_model.model.set_params(union__question_bow__tfidf=None)

    # dimension reduction
    if reduce_dim == 'None':
        sms_guru_model.model.set_params(union__question_bow__reduce_dim=None)
    elif reduce_dim == 'trunSVD':
        sms_guru_model.model.set_params(
            union__question_bow__reduce_dim=TruncatedSVD(n_components=dim))
    elif reduce_dim == 'chi2':
        sms_guru_model.model.set_params(
            union__question_bow__reduce_dim=SelectKBest(chi2, k=dim))

#     # get features
    features = sms_guru_model.fit_transform()
#     print(model)
#     # get feature names
#     if reduce_dim == 'None':
#         featurenames = model.named_steps['union'].transformer_list[0][1].named_steps['vectorize'].get_feature_names()
#         featurenames = np.asarray(featurenames)
#     elif reduce_dim == 'trunSVD':
#         # no interpretable feature names
#         featurenames = None
#     elif reduce_dim == 'chi2':
#         print(model.named_steps_)
#         featurenames = np.asarray(
#             model.named_steps[
#                 'union__question_bow__vectorize'].get_feature_names())
#         featurenames = featurenames[
#             model.named_steps['union__question_bow__reduce_dim'].get_support()]
# 
    # add meta data label
    featurenames = sms_guru_model.get_feature_names()
    if metadata and featurenames is not None:
        metadata_label = sms_guru_model.model.named_steps['union'].transformer_list[1][1].named_steps['vectorize'].get_feature_names()
        featurenames = np.asarray(featurenames.tolist() + metadata_label)

    if verbose:
        print("feature matrix size {}".format(features.T.shape))
        print("featurenames size {}".format(len(featurenames)))
        print("categoryids size {}".format(
            len(sms_guru_model.question_loader_.categoryids)))
        print("categories size: {}".format(
            len(sms_guru_model.question_loader_.categories)))
        print("number of questions: {}".format(
            len(sms_guru_model.question_loader_.questions)))
        print("filtered because of min_df = {}:".format(min_df))
        # print(model.named_steps['vectorize'].stop_words_)
        print("feature names: {}".format(featurenames))
    # save extracted features
    np.savez(outfile, features=features.T.toarray(),
             featurenames=featurenames,
             categoryids=sms_guru_model.question_loader_.categoryids[None, :],
             categories=sms_guru_model.question_loader_.categories)


# run extract_features method if module is executed as a script
# put non-default input here in function
if __name__ == "__main__":
    # extract_features(tokenizer='word_tokenizer',
    #                  tfidf=True, mapnumbers=True, min_df=2)
    plac.call(extract_features)
