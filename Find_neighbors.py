import numpy as np
from Parse_POSCAR import parse_POSCAR
from atoms_dist import atoms_dist
import math

def find_neighbors(ctrAtom, cutoff, POSCAR="POSCAR"):
    """ Return the coordinates of neighboring atoms.

    Use VASP 5 format POSCAR or CONTCAR files. Periodic boundary
    conditions are taken into accont of.

    Arguments:
    -------------------
    ctrAtom : str
        Species and index of the central atom. e.g.: "O12", "Fe3".

    cutoff : float
        The cutoff radius, centered at ctrAtom (in angstrom).
        All atoms within cutoff radius of the center atom are counted.

    POSCAR : str
        Input file name. Must be VASP 5 format.

    Returns:
    -------------------
    res : dict {'str': dict{str : list[float]} }
        Each key represents one atomic species. Each value is
        a dict {"element + index" : coordinates}.
    """

    # center atom species and index
    #ctrAtom_name = ''.join(i for i in ctrAtom if i.isalpha())
    #ctrAtom_index = int(''.join(i for i in ctrAtom if i.isdigit()))
    if not ctrAtom[1].isalpha():
        ctrAtom_name = ctrAtom[0]
        ctrAtom_index = int(ctrAtom[1:])
    else:
        ctrAtom_name = ctrAtom[0:2]
        ctrAtom_index = int(ctrAtom[2:])

    # extract lattice constants and atom coordinates dictionary using POSCAR parser
    latt_mat, latt_consts, _, _, atomCoor_Dict = parse_POSCAR(POSCAR)

    # coordinates of the central atom
    ctrAtomCoor = atomCoor_Dict[ctrAtom_name][ctrAtom_index - 1].reshape((1, 3))
    length_a, length_b, length_c = latt_consts
    # calculate distances to the central atom
    res = dict.fromkeys(atomCoor_Dict, {})
    for i in atomCoor_Dict:
        res[i] = {}  # avoid name-binding problem!
        for coor in atomCoor_Dict[i]:
            currCoor = coor.reshape((1, 3))
            # dist_base = atoms_dist(ctrAtomCoor, currCoor, latt_consts) # distance within the simulation cell
            index = np.where(np.all(atomCoor_Dict[i] == coor, axis=1))[0][0]
            # need to consider the periodic boundary condition
            repetition_a = math.ceil(cutoff / length_a)  # upper boundary of number of adjacent cells to search
            repetition_b = math.ceil(cutoff / length_b)
            repetition_c = math.ceil(cutoff / length_c)
            # usually a, b and c are not big values, so this nested loop does not take much time
            for a in range(-repetition_a, repetition_a + 1):
                for b in range(-repetition_b, repetition_b + 1):
                    for c in range(-repetition_c, repetition_c + 1):
                        dist_to = currCoor + np.array([[a, 0, 0]]) + np.array([[0, b, 0]]) + np.array([[0, 0, c]])
                        if atoms_dist(ctrAtomCoor, dist_to, latt_mat) <= cutoff:
                            res[i][i + str(int(index + 1))] = list(np.squeeze(dist_to))
    # remove the central atom from the dictionary
    del res[ctrAtom_name][ctrAtom]
    return(res)

#print(find_neighbors("O12", 2.6, "/Users/mdlu8/Dropbox (MIT)/Python/argonne/POSCAR"))
