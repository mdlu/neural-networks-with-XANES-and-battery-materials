import numpy as np
import math


def parse_POSCAR(POSCAR="POSCAR"):
    """ Parse the POSCAR (or CONTCAR) file to extract crystal strucure
        information. Currently only support VASP 5 format.
        Return lattice matrix, lattic constants, angles between lattice vectors,
        dictionary of atom numbers and dictionary of atomc coordinates.

    Arguments:
    -------------------
    POSCAR : str
        Input file name. Must be VASP 5 format.

    Returns:
    -------------------
    latt_mat : array(float), dim = (3, 3)
        Matrix consisting of lacttice vectors a, b and c.

    latt_consts : list[float], dim = (1, 3)
        a, b, c directions lattice constants of the cell.

    angles : list[float], dim = (1, 3)
        alpa (between b and c), beta (between a and c) and gamma (between a and b)
        crystal angles (deg).

    atomNum_Dict : dict['str': int]
        zip atomNames and atomNums to form the dictionary.
        Each key represents one atomic species.

    atomCoor_Dict : dict['str': 2D array]
        Each key represents one atomic species. Values are 2D arrays
        of atomic coordinates. Dimension of 2D array is contingent to atomNums.
    """

    fin = open(POSCAR, 'r')
    poscar = fin.read().splitlines()
    scaling_para = float(poscar[1])
    abc = np.array([[float(i) for i in line.split()] for line in poscar[2:5]])

    # lattice constants in angstrom
    latt_mat = abc * scaling_para
    length_a = np.linalg.norm(latt_mat[0, :], 2)
    length_b = np.linalg.norm(latt_mat[1, :], 2)
    length_c = np.linalg.norm(latt_mat[2, :], 2)
    latt_consts = [length_a, length_b, length_c]

    # angles in degrees
    alpha = angle_btw(abc[1, :], abc[2, :])
    beta = angle_btw(abc[0, :], abc[2, :])
    gamma = angle_btw(abc[0, :], abc[1, :])
    angles = [alpha, beta, gamma]

    # Lines 6 and 7 of POSCAR. atomic species and corresponding atoms numbers
    atomNames = poscar[5].split()
    atomNums = list(map(int, poscar[6].split()))
    # combine atom names and numbers into a dict
    atomNum_Dict = dict(zip(atomNames, atomNums))
    # read in the coordinates of each species
    atomCoor_Dict = dict.fromkeys(atomNum_Dict, [])
    st_line = 8  # starting line number of atom coordinates
    for i in atomCoor_Dict.keys():
        end_line = st_line + atomNum_Dict[i]
        coor = np.array([[float(e) for e in line.split()[0:3]] for line in poscar[st_line: end_line]])
        st_line = end_line
        atomCoor_Dict[i] = coor
    fin.close()

    return latt_mat, latt_consts, angles, atomNum_Dict, atomCoor_Dict


def angle_btw(v1, v2):
    """ Return the angle between vectors v1 and v2 in degrees.
    """
    cos_ang = np.dot(v1, v2)
    sin_ang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sin_ang, cos_ang) * 180 / math.pi

# latt_mat, latt_consts, angles, atomNum_Dict, atomCoor_Dict = parse_POSCAR("/Users/mdlu8/Dropbox (MIT)/Python/argonne/POSCAR")
# print(latt_mat)
# print(latt_consts)
# print(angles)
# print(atomNum_Dict)
# print(atomCoor_Dict)