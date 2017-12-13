"""
The :mod: `evaluation` implements some helper functions for performing
and saving the gridsearch results
"""
# Author: Ingo Gühring

from time import gmtime, strftime
import numpy as np


# number of folds used in cross-validation
CV = 3

# in milestone it suffices to focus on parent categories
subcats = False


def merge_two_dicts(x, y):
    """Return merged dict"""
    z = x.copy()
    z.update(y)
    return z


def save_and_report(results, folder, report=None):
    """Save results and info about the performed search"""

    # save results
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    path = './results/gridsearch/' + folder + '/'
    filename = current_time + 'grids_cv.npy'
    np.save(path + filename, results)

    if report is not None:
        # update report
        with open(path + "/gridsearches.txt", "a") as report:
            report.write("performed at: {}, non_grid_params:  "
                         .format(current_time) + report)
