import numpy as np
import os
import sys
import math
sys.path.append("/Users/mdlu8/Dropbox (MIT)/Python/argonne/")
from atoms_dist import atoms_dist
from Find_neighbors import find_neighbors


def extract_training_data(num = 5000, cutoff_radius = 2.6, augment = False):
    """ Calculate Fe coordination numbers around oxygen. Return trainin, dev, and test sets.
        Fourteen Li3FeO3.5 POSCARs are used. 170 data points for each XANES spectrum.
        If calculating the average coordination number by averaging 49 spectra together, augment is sent to True, and we use regression.

    Arguments:
    ----------------------------
    num: the number of combinations of 49 spectra desired 
    cutoff_radius: the maximum length, in angstroms, an Fe atom can be from an oxygen such that it contributes to the oxygen's Fe coordination number
    augment: signals whether to use data augmentation to generate data by averaging 49 spectra at a time

    Returns:
    ----------------------------
    shuffled_X[:, :divider1] : array (170, 64% of data)
        Training set input

    shuffled_X[:, divider1:divider2] : array (170, 16% of data)
        Dev set input

    shuffled_X[:, divider2:] : array (170, 20% of data)
        Test set input

    shuffled_Y[:, :divider1] : array (1, 64% of data)
        Training set output

    shuffled_Y[:, divider1:divider2] : array (1, 16% of data)
        Dev set output
    
    shuffled_Y[:, divider2:] : array (1, 20% of data)
        Test set output

    numOutputNodes : int
        Number of nodes of output layer, max(Fe coordination number) + 1

    """
    X = np.zeros((170, 1), float)  # 170 is the number of data points in the spectrum
    Y = []

    # extract the data, stored in three separate directories
    for i in ['0K', '15ps', '20ps']:
        prefix = "/Users/lum/Downloads/argonne/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + i + "_Combo_O_all.dat")
        # energy range [-1 eV ~ 14 eV], 170 data points
        X = np.concatenate((X, tmp[436:606, 1:]), 1)
        for O_index in range(1, 50): # cycles through the 49 oxygen atoms present
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + "POSCAR_" + i)
            FeCoorNum.append(len(res['Fe'])) # this adds the calculated Fe coordination number for this particular oxygen atom
        Y = Y + FeCoorNum
    
    for i in ['300K/', '1000K/']:
        for n in ['01/', '02/', '03/', '04/', '05/']:
            prefix = "/Users/lum/Downloads/argonne/new_data/" 
            FeCoorNum = []
            tmp = np.loadtxt(prefix + i + n + "Combo_O_all.dat")
            X = np.concatenate((X, tmp[438:608, 1:]), 1)
            for O_index in range(1, 50):
                res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + i + n + "CONTCAR")
                FeCoorNum.append(len(res['Fe']))
            Y = Y + FeCoorNum

    prefix = "/Users/lum/Downloads/argonne/new_data/300K/06/" 
    FeCoorNum = []
    tmp = np.loadtxt(prefix + "Combo_O_all.dat")
    X = np.concatenate((X, tmp[438:608, 1:]), 1)
    for O_index in range(1, 50):
        res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + "CONTCAR")
        FeCoorNum.append(len(res['Fe']))
    Y = Y + FeCoorNum
    

    X = np.delete(X, 0, 1) # remove the first column of zeros used to initialize X
    
    if augment:
        X, Y = data_augmentation(X, Y, num) # computes averaged spectra and labels

    Y = np.array(Y).reshape(1, X.shape[1])
    numOutputNodes = int(Y.max()) + 1

    # shuffle the input data
    m = X.shape[1]
    # np.random.seed(0) # used if consistency is desired
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # generate training, development, and test sets (64:16:20 ratio)
    divider1 = math.floor(m*16/25)
    divider2 = math.floor(m*4/5)
    
    # original preprocessing 
    # X = X * 1e7 / 6
    
    # new, updated preprocessing: compute the mean and stdev for only the training data, and subtract from the whole data set
    mu = np.mean(shuffled_X[:, :divider1], axis=1).reshape(170, 1) 
    std = np.std(shuffled_X[:, :divider1], axis=1).reshape(170, 1)
    shuffled_X = (shuffled_X - mu) / std
    
    return shuffled_X[:, :divider1], shuffled_X[:, divider1:divider2], shuffled_X[:, divider2:], shuffled_Y[:, :divider1], \
    shuffled_Y[:, divider1:divider2], shuffled_Y[:, divider2:], numOutputNodes


def data_augmentation(X, Y, num):
    """ Given data set X and its corresponding labels Y, augments the data by averaging 49 spectra  
        and labels together.

        Arguments:
        ----------------------------
        X, Y: the original spectra and corresponding labels
        num: number of new spectra desired (bounded by the number of examples, choose 49)

        Returns:
        ----------------------------
        bigX, bigY: augmented versions of X and Y
    """
    bigX = np.zeros((170, 1), float)
    bigY = np.zeros((5, 1), float) # the labels can be 0, 1, 2, 3, or 4; this allows us to keep track of how many spectra of each individual label are used

    Y = np.array(Y).reshape(1, len(Y))
    Y_one_hot = np.zeros((5, Y.size)) # creates a one-hot vector with Y
    Y_one_hot[Y.astype(int), np.arange(Y.size)] = 1

    for n in range(num): # repeat to get a 'num' number of samples
        indices = np.random.choice(Y.size, 49, replace = False) # randomly selects 49 columns to use, without replacement
        chooseX = X[:, indices]
        chooseY = Y_one_hot[:, indices]
        newX = np.sum(chooseX, axis=1).reshape((170, 1)) / 49 # averages 49 spectra
        newY = np.sum(chooseY, axis=1).reshape((5, 1)) / 49 # averages 49 labels
        bigX = np.concatenate((bigX, newX), 1)
        bigY = np.concatenate((bigY, newY), 1)

    # remove the first column of zeros that we used to initialize bigX and bigY
    bigX = np.delete(bigX, 0, 1) 
    bigY = np.delete(bigY, 0, 1)

    weights = np.array([0,1,2,3,4])
    bigY = np.matmul(weights, bigY).reshape((1, num)) # finds the weighted average of all labels

    return bigX, bigY