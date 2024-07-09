from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from itertools import combinations
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import csv
import argparse
from distutils.util import strtobool
from math import ceil


def get_site_bond_angles(struc, idx, cnn):
    """
    Gets the bond angles of a site with all of its nearest neighbors.

    Parameters
    ----------
    struc : Pymatgen Structure object
    idx : site index
    cnn : CrystalNN instance

    Returns
    -------
    bond_angles: list of bond angles

    """
    # Find nearest neighbors of the indexed site
    nn_all = cnn.get_nn_info(struc, idx)
    if len(nn_all) > 0:
        nn_coord_all = [nn['site'].coords for nn in nn_all]
        center_coord = struc[idx].coords
    
        # Calculate bond angles
        bond_angles = []
        for comb in combinations(np.arange(len(nn_coord_all)), 2): 
            u = nn_coord_all[comb[0]] - center_coord
            v = nn_coord_all[comb[1]] - center_coord
            cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            if np.isnan(np.arccos(cos_angle)):   # Range outside of [-1, 1] due to numerical error
                angle = np.arccos(round(cos_angle, 5))
            else:
                angle = np.arccos(cos_angle)
            angle = angle * 180 / np.pi   # Convert to degrees
            bond_angles.append(angle)
    else:   # No nearest neighbors found
        bond_angles = []
    
    return bond_angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split run into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_relaxed.extxyz', help='path to data (extxyz file)')
    
    args = parser.parse_args()
    batch = args.batch
    
    if args.split_batches:
        data = read(filename='../'+args.data_path, index=':', format='extxyz')
    else:
        data = read(filename=args.data_path, index=':', format='extxyz')
    csv_fname = 'bond_angles.csv'   # For saving results
    
    # Write csv header
    with open(csv_fname, "w", newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['structure_number', 'bond_angle_list'])
    
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
    
    # Get bond angles
    print('Getting bond angles')
    ase_adaptor = AseAtomsAdaptor()
    cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
    
    for i in tqdm(range(i_start, i_end+1)):
        atoms = data[i]
        struc = ase_adaptor.get_structure(atoms)
        
        bond_angles_all = []   # Stores the bond angles iterated over all sites
        atom_indices = np.arange(len(struc))
        for j in atom_indices:
            bond_angles = get_site_bond_angles(struc, j, cnn)
            bond_angles_all += bond_angles
        with open(csv_fname, 'a', newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([i+1] + [bond_angles_all])


if  __name__ == '__main__':
    main()
