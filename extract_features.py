# -*- coding: UTF-8 -*-
# import numpy as np
from pprint import pprint
import create_model
import question_loader as ql


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     countwords=True,
                     metadata=True,
                     spellcorrection=False,
                     stemming=True,
                     subcats=True,
                     tokenization='word_tokenize',
                     outfile='features.npz'):

    loader = ql.QuestionLoader(qfile=qfile, catfile=catfile, subcats=subcats)
    model = create_model.create_pipeline()
    model.set_params(reduction=None)
    features = model.fit_transform(loader.questions, loader.categoryids)
    # featurenames = model.vocabulary_
    featurenames = model.named_steps['vectorize'].get_feature_names()
    print("feature size{}".format(features.shape))
    print("featurenames size {}".format(len(featurenames)))
    print("categoryids size {}".format(len(loader.categoryids)))
    print("categories size: {}".format(len(loader.categories)))
    print("number of questions: {}".format(len(loader.questions)))
    print(loader.categories)
    print(loader.subcats)
    return None


# run extract_features method if module is executed as a script
# put non-default input here in function
if __name__ == "__main__":
    extract_features()
