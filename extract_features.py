# -*- coding: UTF-8 -*-
# import numpy as np
import create_model
import question_loader as ql
import numpy as np
import pprint


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     binary=False,
                     countwords=True,
                     mapnumerics=True,
                     metadata=True,
                     spellcorrector=False,
                     stemmer=True,
                     subcats=True,
                     reduction=False,
                     tfidf=False,
                     tokenizer='word_tokenizer',
                     outfile='features.npz',
                     verbose=True):

    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile,
                               subcats=subcats, verbose=verbose)
    model = create_model.create_pipeline()
    # tokens is the name of the first transformation in the pipeline
    model.set_params(tokens__mapnumerics=mapnumerics,
                     tokens__spellcorrector=spellcorrector,
                     tokens__stemmer=stemmer,
                     tokens__tokenizer=tokenizer,
                     vectorize__binary=binary,
                     )
    if not tfidf:
        model.set_params(tfidf=None)
    if not reduction:
        model.set_params(reduction=None)
    features = model.fit_transform(loader.questions, loader.categoryids)
    featurenames = model.named_steps['vectorize'].get_feature_names()
    if verbose:
        print("feature matrix size {}".format(features.T.shape))
        print("featurenames size {}".format(len(featurenames)))
        print("categoryids size {}".format(len(loader.categoryids)))
        print("categories size: {}".format(len(loader.categories)))
        print("number of questions: {}".format(len(loader.questions)))
    # save extracted features
    np.savez(outfile, features=features.T,
             featurenames=featurenames,
             categoryids=loader.categoryids,
             categories=loader.categories)
    print(features > 1)


# run extract_features method if module is executed as a script
# put non-default input here in function
if __name__ == "__main__":
    extract_features(tfidf=False)
