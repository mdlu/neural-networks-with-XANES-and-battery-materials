import numpy as np
import os
import sys
import math
sys.path.append("/Users/mdlu8/Dropbox (MIT)/Python/argonne/")
from atoms_dist import atoms_dist
from Find_neighbors import find_neighbors


def extract_training_data(cutoff_radius = 2.6): # update the text below to reflect how many spectra are actually used
    """ Calculate Fe coordination numbers around oxygen. Return training and test sets.
        Three Li3FeO3.5 POSCARs are used. 170 data points for each XANES spectrum.
        49*3 = 147 spectra in total.

    Returns:
    ----------------------------
    shuffled_X[:, :divider1] : array (170, 60% of data)
        Training set input

    shuffled_X[:, divider1:divider2] : array (170, 20% of data)
        Dev set input

    shuffled_X[:, divider2:] : array (170, 20% of data)
        Test set input

    shuffled_Y[:, :divider1] : array (1, 60% of data)
        Training set output

    shuffled_Y[:, divider1:divider2] : array (1, 20% of data)
        Dev set output
    
    shuffled_Y[:, divider2:] : array (1, 20% of data)
        Test set output

    numOutputNodes : int
        Number of nodes of output layer, max(Fe coordination number) + 1

    """
    X = np.zeros((170, 1), float)  # number of data points in the spectrum
    Y = []
    for i in ['0K', '15ps', '20ps']:
        prefix = "/Users/mdlu8/Dropbox (MIT)/Python/argonne/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + i + "_Combo_O_all.dat")
        # energy range [-1 eV ~ 14 eV], 170 data points
        X = np.concatenate((X, tmp[436:606, 1:]), 1)
        for O_index in range(1, 50):
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + "/POSCAR_" + i)
            FeCoorNum.append(len(res['Fe']))
        Y = Y + FeCoorNum
    X = np.delete(X, 0, 1)
    X = X * 1e7 / 6
    Y = np.array(Y).reshape(1, X.shape[1])
    numOutputNodes = int(Y.max()) + 1

    # shuffle the input data
    m = X.shape[1]
    np.random.seed(0)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # generate training, development, and test sets (60:20:20 ratio)
    divider1 = math.floor(m*3/5)
    divider2 = math.floor(m*4/5)
    return shuffled_X[:, :divider1], shuffled_X[:, divider1:divider2], shuffled_X[:, divider2:], shuffled_Y[:, :divider1], \
    shuffled_Y[:, divider1:divider2], shuffled_Y[:, divider2:], numOutputNodes