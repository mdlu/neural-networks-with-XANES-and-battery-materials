import numpy as np


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
