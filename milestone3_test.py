#!/usr/bin/python
# -*- coding: utf-8 -*-
# Test for milestone 3 of the Machine Learning Project.
# Author: Jan Saputra Mueller, Daniel Bartz

import csv
import sys
import os
import numpy
import math

from predict_question_category import predict_question_category

def milestone3_test(qfile = 'question_train_subsample.csv', qcatfile = 'question_category_train_subsample.csv', catfile = 'category.csv'):
  ''' Test for milestone 3. This test uses your classifier to predict categories for a test data set,
      and does some evaluation.

     input
       qfile     file with test questions
       qcatfile  file with categories for the test questions (needed for evaluation)
       catfile   file with categories
  '''

  # read table category
  file_category = open(catfile, 'r')
  reader_category = csv.reader(file_category, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  major_categories = {} # major categories dictionary: category id -> major category name
  minor_categories = {} # minor categories dictionary: category id -> minor category name
  minor_major = {} # dictionary: minor category -> major category
  next(reader_category) # skip first line
  for line in reader_category:
    if line[1] == '0':
      # category is a major category
      major_categories[int(line[0])] = line[2]
    else:
      # category is a minor category
      minor_categories[int(line[0])] = line[2]
      minor_major[int(line[0])] = int(line[1])
  file_category.close()

  # get true "question -> minor category" labeling
  file_qc = open(qcatfile, 'r')
  reader_qc = csv.reader(file_qc, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  qc = {} # map: question id -> category id
  question_category_columns = next(reader_qc) # first line contains the column names
  for line in reader_qc:
    qc[int(line[2])] = int(line[1])
  file_qc.close()

  # read questions
  file_question = open(qfile, 'r')
  reader_question = csv.reader(file_question, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  major_labels = [] # true major labels
  minor_labels = [] # true minor labels
  next(reader_question) # first line contains the column names
  for line in reader_question:
    qid = int(line[0])
    if qid in qc:
      label = qc[qid]
      minor_labels.append(label)
      major_labels.append(minor_major[label])
    else:
      # question has no category
      minor_labels.append(-1)
      major_labels.append(-1)
  file_question.close()

  # make predictions
  predictions = predict_question_category(qfile)

  # print confidences
  print_prediction_confidence(predictions, 0.95, major_categories, minor_categories, major_labels, minor_labels)
  print_prediction_confidence(predictions, 0.75, major_categories, minor_categories, major_labels, minor_labels)
  print_prediction_confidence(predictions, 0.5, major_categories, minor_categories, major_labels, minor_labels)
  print_prediction_confidence(predictions, 0.0, major_categories, minor_categories, major_labels, minor_labels)


def print_prediction_confidence(predictions, confidence_level, major_categories, minor_categories, major_labels, minor_labels):
  ''' Prints out some information about the prediction confidence '''

  sys.stdout.write('==== Confidence level: ' + percent1(confidence_level) + '\n')
  # get questions with confidence > confidence_level
  #confident_predictions_major = filter(lambda x: x['confidence_major_cat'] > confidence_level, predictions)
  major = [x['confidence_major_cat'] > confidence_level for x in predictions]
  #confident_predictions_minor = filter(lambda x: x['confidence_minor_cat'] > confidence_level, predictions)
  minor = [x['confidence_minor_cat'] > confidence_level for x in predictions]
  sys.stdout.write('Number of classified data points (major classes): ' + str(positives(major)) + '/' + str(len(predictions)) + '\n')
  sys.stdout.write('Number of classified data points (minor classes): ' + str(positives(minor)) + '/' + str(len(predictions)) + '\n')
  major_predictions_filtered = filter_mask(predictions, major)
  minor_predictions_filtered = filter_mask(predictions, minor)
  major_labels_filtered = filter_mask(major_labels, major)
  minor_labels_filtered = filter_mask(minor_labels, minor)
  (mean, std) = mean_std([x['confidence_major_cat'] for x in major_predictions_filtered])
  sys.stdout.write('Confidence major classes: mean=' + percent1(mean) + ' std=' + percent1(std) + '\n')
  (mean, std) = mean_std([x['confidence_minor_cat'] for x in minor_predictions_filtered])
  sys.stdout.write('Confidence minor classes: mean=' + percent1(mean) + ' std=' + percent1(std) + '\n')

  # major categories
  sys.stdout.write('\n== Errors on major classes\n')
  major_predictions_ids = [x['major_category'] for x in major_predictions_filtered]
  for category in major_categories:
    sys.stdout.write(major_categories[category]
                        + align_names(len(major_categories[category]), 30)
                        + ': FPR=' + percent1(fpr(major_predictions_ids, major_labels_filtered, category))
                        + ' TPR=' + percent1(tpr(major_predictions_ids, major_labels_filtered, category)) + '\n')

  # minor categories
  sys.stdout.write('\n== Errors on minor classes\n')
  minor_predictions_ids = [x['minor_category'] for x in minor_predictions_filtered]
  for category in minor_categories:
    sys.stdout.write(minor_categories[category]
                        + align_names(len(minor_categories[category]), 30)
                        + ': FPR=' + percent1(fpr(minor_predictions_ids, minor_labels_filtered, category))
                        + ' TPR=' + percent1(tpr(minor_predictions_ids, minor_labels_filtered, category)) + '\n')

  sys.stdout.write('\n')
  # ipdb.set_trace()
  sys.stdout.write('accuracy major: '
                    + percent1(accu(  major_predictions_ids, major_labels_filtered )) + '\n')
  sys.stdout.write('accuracy minor: '
                    + percent1(accu(  minor_predictions_ids, minor_labels_filtered )) + '\n')
  # ipdb.set_trace()
  sys.stdout.write('\n')

def mean_std(l):
  ''' Returns the mean and std for the elements in l '''
  mean = numpy.mean(list(l))  #reduce(lambda x, y: x + y, l) / float(len(l))
  std = numpy.std(list(l))  #math.sqrt(reduce(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, l)) / (float(len(l)) - 1.0))
  return(mean, std)

def positives(l):
  ''' Counts how often "true" appears in the list l '''
  return numpy.sum(l)  #reduce(lambda x, y: x + y, map(lambda x: 1 if x else 0, l))

def filter_mask(l, m):
  ''' Filters the elements of l according to the mask m (list containing booleans) '''
  # ipdb.set_trace()
  return [l[i] for i in range(len(l)) if m[i]]  #map(lambda x: x[1], filter(lambda x: x[0], zip(m, l)))

def tpr(predictions, true_labels, category):
  ''' Calculates the true positive rate for a category '''
  category_mask = [x == category for x in true_labels]
  true_positives = [x == category for x in filter_mask(predictions, category_mask)]
  n_positive = positives(category_mask)
  if n_positive > 0:
    return float(positives(true_positives)) / float(n_positive)
  else:
    return None

def fpr(predictions, true_labels, category):
  ''' Calculates the false positive rate for a category '''
  not_category_mask = [x != category for x in true_labels]
  false_positives = [x == category for x in filter_mask(predictions, not_category_mask)]
  n_negatives = positives(not_category_mask)
  if n_negatives > 0:
    return float(positives(false_positives)) / float(n_negatives)
  else:
    return None

def percent1(f):
  if f is None:
    return '  None'
  else:
    s = ('%.1f' % (100*f)) + '%'
    s = align_names(len(s), 6) + s
    return s

def accu(predictions, true_labels):
    ''' returns the ratio of correct predictions '''
    return numpy.mean([predictions[i] == true_labels[i] for i in range(len(true_labels))])
    #l = zip(true_labels, predictions)
    #return reduce(lambda x, y: x + y,
    #                 map(lambda x: 1 if x[0] == x[1] else 0, l) )/float(len(l))

def blanks(n):
  s = ''
  for i in range(n):
    s += ' '
  return s

def align_names(n, max_l):
  return blanks(max_l - n)

# ***************************** main program *******************************************
if __name__ == "__main__":
  milestone3_test()
