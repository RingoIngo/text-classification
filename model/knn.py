import math
import numpy as np


def cosine_similarity(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def cosine_semi_metric(x, y):
    return 1 - np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def cosine_dist_to_sim(dist):
    return (dist - 1) * (- 1)
