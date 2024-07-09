import count
import symop
from pymatgen.core import Molecule
from pymatgen.symmetry import analyzer as syman
import json
import numpy as np
import copy
import argparse
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from distutils.util import strtobool
from math import ceil
from tqdm import tqdm
import pandas as pd


def get_rotated_basis_vectors(struc, struc_idx, margin=0.4):
    """
    Gets the basis vectors in the orientation of a rotated FCC structure

    Parameters
    ----------
    struc : Pymatgen Structure object
    struc_idx: Index number of structure
    margin : Bond distance margin of error (unit: Angstrom) for determining 2nd nearest neighbors;
             an Exception will be raised if all six 2nd nearest neighbors are not within margin

    Returns
    -------
    basis_rot: 3x3 NumPy array of basis vectors

    """
    done = False
    atom_idx = 0   # Initialize center atom index for calculating basis vectors
    # Iterate over all atoms as the center atom until basis vectors are obtained 
    while not done:
        if atom_idx >= len(struc):
            raise Exception('Basis vectors cannot be determined for Structure index {}'.format(struc_idx))
        # Get neighbor bond distances up to 2nd nearest neighbors
        nb_all = struc.get_neighbors(struc[atom_idx], 3.6+margin)
        dist_all = []
        for nb in nb_all:
            dist_all.append(struc[atom_idx].distance_from_point(nb.coords))
        dist_all = np.array(dist_all)
        # Filter 2nd nearest neighbors with bond distance between 3.6-margin and 3.6+margin
        idx_filter = np.where(dist_all >= 3.6-margin)[0]
        if len(idx_filter) == 6:   # Correct number of 2nd nearest neighbors, try to get basis vectors
            # Sort 2nd nearest neighbors based on how much its bond distance deviates from 3.6
            dist_2nn = dist_all[idx_filter]
            dist_2nn_error = np.abs(dist_2nn - 3.6)
            idx_filter_sorted = idx_filter[dist_2nn_error.argsort()]
            nb_2nn = [nb_all[i] for i in idx_filter_sorted]
            
            # Get the first a axis
            a_new = nb_2nn[0].coords - struc[atom_idx].coords
            a_new /= np.linalg.norm(a_new)
            # Iterate neighbors to find a perpendicular vector (b axis)
            for idx_nb in range(1, 6):
                b_new = nb_2nn[idx_nb].coords - struc[atom_idx].coords
                b_new /= np.linalg.norm(b_new)
                ab_angle = np.arccos(np.dot(a_new, b_new))
                ab_angle_error = np.abs(ab_angle - np.pi/2)   # Deviation of angle from 90 degrees
                if ab_angle_error <= 0.0175:   # a and b axes are perpendicular enough, accept this basis
                    # Get the third c axis using cross product
                    c_new = np.cross(a_new, b_new)
                    basis_rot = np.array([a_new, b_new, c_new])
                    done = True
                    break
        # Did not get basis vectors using this atom, move on to the next atom
        atom_idx += 1
        
    return basis_rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split run into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_clean.extxyz', help='path to extxyz file of structures')
    parser.add_argument('--lat_in', type=str, default='utils/POSCAR_fcc_unit', help='reference structure for fcc lattice')
    parser.add_argument('--margin', type=float, default=0.8, help='margin of error for distance between designated cluster coordinate and actual site coordinate (unit: Angstrom)')
    
    args = parser.parse_args()
    batch = args.batch
    
    if args.split_batches:
        data = read(filename='../'+args.data_path, index=':', format='extxyz')
    else:
        data = read(filename=args.data_path, index=':', format='extxyz')
    
    # Get starting and ending indexes of structures to analyze
    if args.split_batches:
        n_strucs_per_batch = args.n_strucs_per_batch
        i_start = (batch - 1)*n_strucs_per_batch   # Starting index
        if batch == ceil(len(data) / n_strucs_per_batch):   # last batch may have less than n_strucs_per_batch
            i_end = len(data) - 1   # Ending index
        else:
            i_end = batch*n_strucs_per_batch - 1   # Ending index
    else:
        i_start = 0
        i_end = len(data) - 1 
    
    # Get all symmetry-equivalent clusters
    clust_list = [
                    [ [ [0, 0, 0] ], [0], [0] ],
                    [ [ [0, 0, 0], [0.5, 0.5, 0] ], [2.545584], [0] ],
                    [ [ [0, 0, 0], [1, 0, 0] ], [3.6], [0] ],
                    [ [ [0, 0, 0], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [1.5, 0.5, 0] ], [5.6921], [0] ],
                    [ [ [0, 0, 0], [1, 1, 1] ], [6.235383], [0] ],
                    [ [ [0, 0, 0], [1.5, 1, 0.5] ], [6.734983], [0] ],
                    [ [ [0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5] ], [2.545584], [0] ],
                    [ [ [0, 0, 0], [0.5, 0.5, 0], [1, 0, 0] ], [3.6], [0]],
                    [ [ [0, 0, 0], [0.5, 0, 0.5], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [0, 0.5, 0.5], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [0.5, -0.5, 0], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [0, 1, 0], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [0.5, -0.5, 1], [1, 0.5, 0.5] ], [4.409082], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0], [0.5, 0.5, 0] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0], [1, 0.5, 0.5] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0], [1, 0, 0] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0], [0.5, 0.5, 1] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [1, 1, 0], [1, 0, 1] ], [5.091169], [0] ],
                    [ [ [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5] ], [2.54558], [0] ]
                 ]
    clust_list = clust_list[:8]   # Use dimers only
    if args.split_batches:
        sym_list = symop.find_sym('../'+args.lat_in)
    else:
        sym_list = symop.find_sym(args.lat_in)
    symeq_clust_list = []
    for orig_clust in clust_list:
        orig_clust = count.scale_clust(orig_clust)  # transform to scaled Cartesian coord
        symeq_clust_list.append(symop.find_eq_clust(sym_list, orig_clust))
    
    # Get point group symmetry of clusters with >= 3 atoms
    spec_pntsym_list = []
    for symeq_clust in symeq_clust_list:
        if len(symeq_clust[0][0]) <= 2:
            spec_pntsym_list.append([[None]] * len(symeq_clust))
        else:
            pntsym_list = []
            for clust in symeq_clust:
                coords = clust[0]
                hypo_molec = Molecule(['H'] * len(coords), coords)
                pntsym = syman.PointGroupAnalyzer(hypo_molec).get_symmetry_operations()
                pntsym_list.append(pntsym)
            spec_pntsym_list.append(pntsym_list)
    
    # Count clusters
    ase_adaptor = AseAtomsAdaptor()
    count_list_all = []
    vac_count_all = []
    for struc_idx in tqdm(range(i_start, i_end+1)):
        struc = ase_adaptor.get_structure(data[struc_idx])
        # Get rotated basis
        basis_rot = get_rotated_basis_vectors(struc, struc_idx)
        # Rotate the cluster coordinates
        symeq_clust_rot_list = copy.deepcopy(symeq_clust_list)
        for i in range(len(symeq_clust_rot_list)):
            symeq_clust = symeq_clust_rot_list[i]
            for j in range(len(symeq_clust)):
                clust = symeq_clust[j]
                coords = np.array(clust[0])
                norms = np.linalg.norm(coords, axis=1)
                coords_rot = np.matmul(coords, basis_rot)
                norms_rot = np.linalg.norm(coords_rot, axis=1)
                # Rescale rotated coordinates (rotated basis is not necessarily fully perpendicular)
                for l in range(len(norms)):
                    if norms[l] > 0:
                        coords_rot[l] = coords_rot[l] / norms_rot[l] * norms[l]
                symeq_clust_rot_list[i][j][0] = coords_rot.tolist()
        # Turn structure into dictionary
        str_dict = {'StrIdx': struc_idx,
                    'LatPnt': struc.cart_coords.tolist(),
                    'LatVec': struc.lattice.matrix.tolist(),
                    'AtomSum': len(struc)
                    }
        
        new_clust_list = []
        for symeq_clust in symeq_clust_rot_list:
            new_clust_list.append(symeq_clust[0])
        
        count_list, has_vac = count.count_clusters(symeq_clust_list, symeq_clust_rot_list, spec_pntsym_list, 
                                                   str_dict, struc, new_clust_list, args.margin)
        count_list_all.append(count_list)
        if has_vac:
            vac_count = count.count_vac(symeq_clust_rot_list, str_dict, struc, margin=1.2)
        else:
            vac_count = 0
        vac_count_all.append(vac_count)
    
    # Save counts
    with open('cluster_count.json', 'w') as filehandle:
        json.dump(count_list_all, filehandle)
    df = pd.DataFrame.from_dict({'structure_index': np.arange(i_start, i_end+1),
                                 'vac_count': vac_count_all})
    df.to_csv('vac_count.csv', index=False)


if  __name__ == '__main__':
    main()
    