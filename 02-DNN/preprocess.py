## DATA PRE-PROCESS FUNCTIONS
import numpy as np


def standardize(x, mean, std):
    return (x-mean)/std


def normalize(x):
    return (x - np.min(x, axis=0)) / np.max(x, axis=0) - np.min(x, axis=0)


def augment_data(data, labels, rate):

    N  = data.shape[0]
    Ni = int(N*rate)

    data_   = data[:Ni]
    labels_ = labels[:Ni]

    x_augmented = []
    y_augmented = []
    for x, y in zip(data_, labels_):
        
        r = np.random.uniform(low=0, high=5, size=1)[0]
        a = np.random.uniform(low=0, high=2*np.pi, size=1)[0]

        x_trasl = [r*np.cos(a), r*np.sin(a)]

        x_augmented.append([x1 + x2 for x1, x2 in zip(x, x_trasl)])  
        y_augmented.append(y)

    return np.array(x_augmented), np.array(y_augmented)
    




