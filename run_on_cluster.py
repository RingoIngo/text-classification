"""
The :mod:`run_on_cluster` module evaluated the different classifiers
"""
# Author: Ingo Guehring
import argparse
import evaluation.lda_svm as lda_svm
import evaluation.svm as svm
import evaluation.lda as lda
import evaluation.qda as qda
import evaluation.knn as knn
import evaluation.knn_b as knn_b


def evaluate(classifier=None, gridsearch=False, gen_error=False, memory=True):
    print('classifier: {}, gridsearch: {}'
          ', gen_error: {}, memory: {}'.format(
              classifier, gridsearch, gen_error, memory))

    if classifier == 'svm':
        svm.evaluate(gridsearch, gen_error, memory)

    elif classifier == 'lda_svm':
        lda_svm.evaluate(gridsearch, gen_error, memory)

    elif classifier == 'knn':
        knn.evaluate(gridsearch, gen_error, memory)

    elif classifier == 'knn_b':
        knn_b.evaluate(gridsearch, gen_error, memory)

    elif classifier == 'lda':
        lda.evaluate(gridsearch, gen_error)

    elif classifier == 'qda':
        qda.evaluate(gridsearch, gen_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate classifier')
    parser.add_argument('-c', '--classifier',
                        choices=['svm', 'lda_svm', 'knn',
                                 'knn_b', 'lda', 'qda'],
                        default=argparse.SUPPRESS)

    parser.add_argument('-gs', '--gridsearch', dest='gridsearch',
                        action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('-ge', '--gen_error', dest='gen_error',
                        action='store_true',
                        default=argparse.SUPPRESS)

    parser.add_argument('-nm', '--no-memory', dest='memory',
                        action='store_false',
                        default=argparse.SUPPRESS)

    evaluate(**vars(parser.parse_args()))
