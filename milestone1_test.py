#!/usr/bin/python
# -*- coding: utf-8 -*-
# Test for milestone 1 of the Machine Learning Project.
# Author: Jan Saputra Mueller

import csv
import sys
import os
import numpy

from extract_features import extract_features

# call extract_features()
sys.stdout.write('Calling function extract_features() with default parameters... ')
extract_features()
sys.stdout.write('done.\n')

# check whether file 'features.npz' exists
sys.stdout.write('Checking whether file \'features.npz\' has been generated... ')
if os.path.exists('features.npz'):
    sys.stdout.write('OK.\n')
else:
    sys.stdout.write('ERROR: file does not exist.\n')
    sys.exit()

# load file 'features.npz'
sys.stdout.write('Loading file \'features.npz\'... ')
data = numpy.load('features.npz')
sys.stdout.write('OK.\n')

# check content of features.npz
sys.stdout.write('Checking content of \'features.npz\'...\n')

# d x n matrix 'features'
if 'features' in data.files:
    sys.stdout.write('Found variable \'features\': ')
    if type(data['features']).__name__ == 'ndarray':
        (d, n) = data['features'].shape
        sys.stdout.write('d = ' + str(d) + ', n = ' + str(n) + '\n')
    else:
        sys.stdout.write('ERROR: \'features\' is not a numpy array! Found type: ' + type(data['features']).__name__ + ', expected type: ndarray\n')
        sys.exit()
else:
    sys.stdout.write('ERROR: variable \'features\' not found.\n')
    sys.exit()

# list of d entries 'featurenames'
if 'featurenames' in data.files:
    sys.stdout.write('Found variable \'featurenames\': ')
    if type(data['featurenames']).__name__ == 'ndarray':
        l = len(data['featurenames'])
        if l == d:
	    sys.stdout.write('Length of \'featurenames\' equals d: OK.\n')
	else:
	    sys.stdout.write('ERROR: List \'featurenames\' has wrong length! Found: ' + str(l) + ', expected: ' + str(d) + '\n')
	    sys.exit()
    else:
        sys.stdout.write('ERROR: \'featurenames\' is not a numpy array! Found type: ' + type(data['featurenames']).__name__ + ', expected type: ndarray\n')
        sys.exit()
else:
    sys.stdout.write('ERROR: variable \'featurenames\' not found.\n')
    sys.exit()

# 1 x n vector 'categoryids'
if 'categoryids' in data.files:
    sys.stdout.write('Found variable \'categoryids\': ')
    if type(data['categoryids']).__name__ == 'ndarray':
        cidsformat = data['categoryids'].shape
        if cidsformat == (1, n):
	    sys.stdout.write('Format is 1 x n: OK.\n')
	else:
	    sys.stdout.write('ERROR: \'categoryids\' has not the correct format! Found: ' + str(cidsformat[0]) + ' x ' + str(cidsformat[1]) + ', expected: 1 x n\n')
	    sys.exit()
    else:
        sys.stdout.write('ERROR: \'categoryids\' is not a numpy array! Found type: ' + type(data['categoryids']).__name__ + ', expected type: ndarray\n')
        sys.exit()
else:
    sys.stdout.write('ERROR: variable \'categoryids\' not found.\n')
    sys.exit()
    
# dictionary 'categories'
if 'categories' in data.files:
    sys.stdout.write('Found variable \'categories\': ')
    if type(data['categories'].item()).__name__ == 'dict':
        sys.stdout.write('Is a dictionary: OK.\n')
    else:
        sys.stdout.write('ERROR: \'categories\' is not a dictionary! Found type: ' + type(data['categories'].item()).__name__ + ', expected type: dict\n')
        sys.exit()
else:
    sys.stdout.write('ERROR: variable \'categories\' not found.\n')
    sys.exit()
