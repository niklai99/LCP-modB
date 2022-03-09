## DATA PRE-PROCESS FUNCTIONS
import numpy as np


def standardize(x, mean, std):
    return (x-mean)/std


def normalize(x):
    return (x - np.min(x, axis=0)) / np.abs(np.max(x, axis=0) - np.min(x, axis=0))





