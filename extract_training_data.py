import numpy as np
import os
import sys
import math
from Find_neighbors import find_neighbors

def atoms_dist(a, b, latt_mat):
    """ Return the distance between atoms a and b.

    Arguments:
    -------------------
    a, b : array or list, dim = (1, 3)
        Coordinates of two atoms in the cell.

    latt_mat : array, dim = (3, 3)
        Matrix consisting of lacttice vectors a, b and c.

    Returns:
    -------------------
    rtype : float
    """
    return np.linalg.norm(np.dot((a - b), latt_mat), 2)


def extract_training_data(num = 5000, cutoff_radius = 2.6, augment = False, multi = False):
    """ Calculate Fe coordination numbers around oxygen. Return training, cross-validation, and test sets.
        Fourteen Li3FeO3.5 POSCARs are used. 170 data points for each XANES spectrum.
        If calculating the average coordination number by averaging 49 spectra together, augment is sent to True, and we use regression.

    Arguments:
    ----------------------------
    num: the number of combinations of 49 spectra desired 
    cutoff_radius: the maximum length, in angstroms, an Fe atom can be from an oxygen such that it contributes to the oxygen's Fe coordination number
    augment: signals whether to use data augmentation to generate data by averaging 49 spectra at a time
    multi: signals whether it's the multi-task learning model, and we need to extract both coordination number and charge data

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
    if multi:
        Y2 = np.zeros((1, 1), float)

    # extract the data, stored in three separate directories
    if not multi: # there is no charge data for these three POSCARs, so do not use if in multi-task learning
        for i in ['0K', '15ps', '20ps']:
            prefix = "./new_data/original/" 
            FeCoorNum = []
            tmp = np.loadtxt(prefix + i + "_Combo_O_all.dat")
            # energy range [-1 eV ~ 14 eV], 170 data points
            X = np.concatenate((X, tmp[436:606, 1:]), 1)
            for O_index in range(1, 50): # cycles through the 49 oxygen atoms present
                res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + "POSCAR_" + i)
                FeCoorNum.append(len(res['Fe'])) # this adds the calculated Fe coordination number for this particular oxygen atom
            Y = Y + FeCoorNum
    
    for n in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
        prefix = "./new_data/300k/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + n + "/Combo_O_all.dat")
        X = np.concatenate((X, tmp[438:608, 1:]), 1)
        for O_index in range(1, 50):
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + n + "/CONTCAR")
            FeCoorNum.append(len(res['Fe']))
        Y = Y + FeCoorNum
        
        if multi:
            charges = np.loadtxt(prefix + n + "/charge.dat").reshape(1, 49)
            Y2 = np.concatenate((Y2, charges), 1)


    for n in ['01', '02', '03', '04', '05']:
        prefix = "./new_data/1000K/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + n + "/Combo_O_all.dat")
        X = np.concatenate((X, tmp[438:608, 1:]), 1)
        for O_index in range(1, 50):
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + n + "/CONTCAR")
            FeCoorNum.append(len(res['Fe']))
        Y = Y + FeCoorNum
        
        if multi:
            charges = np.loadtxt(prefix + n + "/charge.dat").reshape(1, 49)
            Y2 = np.concatenate((Y2, charges), 1)


    X = np.delete(X, 0, 1) # remove the first column of zeros used to initialize X

    if augment:
        X, Y = data_augmentation(X, Y, num) # computes averaged spectra and labels

    Y = np.array(Y).reshape(1, X.shape[1])
    if multi: 
        Y2 = np.delete(Y2, 0, 1) 
        Y2 = np.array(Y2).reshape(1, X.shape[1])

    if augment:
        numOutputNodes = 1
    else:
        numOutputNodes = int(Y.max()) + 1

    # shuffle the input data
    m = X.shape[1]
    np.random.seed(0) # used if consistency is desired
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # generate training, development, and test sets (64:16:20 ratio)
    divider1 = math.floor(m*16/25)
    divider2 = math.floor(m*4/5)
    
    # normalization: compute the mean and stdev for only the training data, and subtract from the whole data set
    mu = np.mean(shuffled_X[:, :divider1], axis=1).reshape(170, 1) 
    std = np.std(shuffled_X[:, :divider1], axis=1).reshape(170, 1)
    shuffled_X = (shuffled_X - mu) / std
    
    if multi:
        shuffled_Y2 = Y2[:, permutation]

        return shuffled_X[:, :divider1], shuffled_X[:, divider1:divider2], shuffled_X[:, divider2:], shuffled_Y[:, :divider1], \
        shuffled_Y[:, divider1:divider2], shuffled_Y[:, divider2:], shuffled_Y2[:, :divider1], shuffled_Y2[:, divider1:divider2], \
        shuffled_Y2[:, divider2:], numOutputNodes
    
    else:
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

    np.random.seed(0) # for consistency
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


def real_averaged_spectra(num = 10000, cutoff_radius = 2.6):
    """ Returns the average spectrum and coordination number from each POSCAR.
        The 49 spectra are all taken from the same POSCAR file, rather than in data_augmentation, where a set of random 49 are taken. """

    X = np.zeros((170, 1), float)  # 170 is the number of data points in the spectrum
    Y = []
    
    X_avg = np.zeros((170, 1), float)
    Y_avg = np.zeros((5, 1), float)

    # extract the data, stored in three separate directories
    for i in ['0K', '15ps', '20ps']:
        prefix = "./new_data/original/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + i + "_Combo_O_all.dat")
        # energy range [-1 eV ~ 14 eV], 170 data points
        X = np.concatenate((X, tmp[436:606, 1:]), 1)
        for O_index in range(1, 50): # cycles through the 49 oxygen atoms present
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + "POSCAR_" + i)
            FeCoorNum.append(len(res['Fe'])) # this adds the calculated Fe coordination number for this particular oxygen atom
        Y = Y + FeCoorNum
    
    for n in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
        prefix = "./new_data/300k/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + n + "/Combo_O_all.dat")
        X = np.concatenate((X, tmp[438:608, 1:]), 1)
        for O_index in range(1, 50):
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + n + "/CONTCAR")
            FeCoorNum.append(len(res['Fe']))
        Y = Y + FeCoorNum

    for n in ['01', '02', '03', '04', '05']:
        prefix = "./new_data/1000K/" 
        FeCoorNum = []
        tmp = np.loadtxt(prefix + n + "/Combo_O_all.dat")
        X = np.concatenate((X, tmp[438:608, 1:]), 1)
        for O_index in range(1, 50):
            res = find_neighbors('O' + str(O_index), cutoff_radius, prefix + n + "/CONTCAR")
            FeCoorNum.append(len(res['Fe']))
        Y = Y + FeCoorNum

    X_old = np.delete(X, 0, 1) # remove the first column of zeros used to initialize X

    Y_new = np.array(Y).reshape(1, len(Y))
    Y_one_hot = np.zeros((5, Y_new.size)) # creates a one-hot vector with Y
    Y_one_hot[Y_new.astype(int), np.arange(Y_new.size)] = 1

    for i in range(18): # 18, because there are 18 POSCAR/CONTCAR files
        chooseX = X[:, i*49:(i+1)*49]
        chooseY = Y_one_hot[:, i*49:(i+1)*49]
        newX = np.sum(chooseX, axis=1).reshape((170, 1)) / 49 # averages 49 spectra together
        newY = np.sum(chooseY, axis=1).reshape((5, 1)) / 49 # averages 49 labels together
        X_avg = np.concatenate((X_avg, newX), 1)
        Y_avg = np.concatenate((Y_avg, newY), 1)

    # remove the first column of zeros that we used to initialize bigX and bigY
    X_avg = np.delete(X_avg, 0, 1) 
    Y_avg = np.delete(Y_avg, 0, 1)

    weights = np.array([0,1,2,3,4])
    Y_avg = np.matmul(weights, Y_avg).reshape((1, 18)) # finds the weighted average of all labels

    X, _ = data_augmentation(X_old, Y, num) # computes the averaged spectra used in training
    divider1 = math.floor(X.shape[1]*16/25)
    # preprocessing on the real spectra: compute the mean and stdev for only the training data
    # then use these values on the real averaged spectra, because 
    mu = np.mean(X[:, :divider1], axis=1).reshape(170, 1) 
    std = np.std(X[:, :divider1], axis=1).reshape(170, 1)
    X_avg = (X_avg - mu) / std

    return X_avg, Y_avg


# # save the calculated, real averaged spectra
# X_avg, Y_avg = real_averaged_spectra()
# np.save('xrealaveraged.npy', X_avg)
# np.save('yrealaveraged.npy', Y_avg)
# np.set_printoptions(threshold = np.nan)
# print(X_avg)
# print(Y_avg)    

# # save the calculated data to files for fast retrieval
# # used for the classification task of Fe coordination numbers
# a, b, c, d, e, f, g = extract_training_data()
# np.save('./parsed_data/xtraincoords.npy', a)
# np.save('./parsed_data/xdevcoords.npy', b)
# np.save('./parsed_data/xtestcoords.npy', c)
# np.save('./parsed_data/ytraincoords.npy', d)
# np.save('./parsed_data/ydevcoords.npy', e)
# np.save('./parsed_data/ytestcoords.npy', f)

# # used for the regression task on average coordination numbers
# h, i, j, k, l, m, n = extract_training_data(num = 10000, augment = True)
# np.save('./parsed_data/xtrainavgcoords.npy', h)
# np.save('./parsed_data/xdevavgcoords.npy', i)
# np.save('./parsed_data/xtestavgcoords.npy', j)
# np.save('./parsed_data/ytrainavgcoords.npy', k)
# np.save('./parsed_data/ydevavgcoords.npy', l)
# np.save('./parsed_data/ytestavgcoords.npy', m)

# # used for multi-task learning; y1 is coordination numbers, y2 is charge data
# o, p, q, r, s, t, u, v, w, x = extract_training_data(multi = True)
# np.save('./parsed_data/xtrainmulti.npy', o)
# np.save('./parsed_data/xdevmulti.npy', p)
# np.save('./parsed_data/xtestmulti.npy', q)
# np.save('./parsed_data/y1trainmulti.npy', r)
# np.save('./parsed_data/y1devmulti.npy', s)
# np.save('./parsed_data/y1testmulti.npy', t)
# np.save('./parsed_data/y2trainmulti.npy', u)
# np.save('./parsed_data/y2devmulti.npy', v)
# np.save('./parsed_data/y2testmulti.npy', w)