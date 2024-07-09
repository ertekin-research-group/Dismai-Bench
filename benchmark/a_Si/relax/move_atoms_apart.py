import argparse
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
import numpy as np
import os
import csv
import pandas as pd
from ase.io import read, write


def move_atoms_apart(struc, min_dist, i, csv_fname):
    """
    Checks for atoms too close together and moves them apart.
    Logs the number of unique pairs of atoms too close together in a csv file
    
    Parameters
    ----------
    struc: Pymatgen Structure object
    min_dist: minimum separation distance of atoms
    i : index of structure being checked (used when writing the csv file)
    csv_fname: path of the csv file for logging
    
    Returns
    ----------
    new_struc: new Structure object with atoms moved apart
    count: number of unique pairs of atoms too close together
    """
    # Get indexes of atoms too close together
    center_indexes, nbr_indexes, _ , _ = struc.get_neighbor_list(min_dist)
    too_close_pair_idx = np.column_stack((center_indexes, nbr_indexes))
    if too_close_pair_idx.size == 0:
        # No atoms too close together
        count = 0
    else:
        # Get unique pairs of atoms too close together
        too_close_pair_idx = np.sort(too_close_pair_idx, axis=1)
        too_close_pair_idx = np.unique(too_close_pair_idx, axis=0)
        too_close_pair_idx = too_close_pair_idx.tolist()
        count = len(too_close_pair_idx)

    # Log the count and indexes of unique pairs of atoms too close together
    # (only logs during the first iteration)
    if i == 0:
        if not os.path.exists(csv_fname):
            with open(csv_fname, "w", newline='') as csvfile: 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['No','count', 'pair_index'])
                csvwriter.writerow([i+1, count, too_close_pair_idx])
    else:
        df = pd.read_csv(csv_fname)
        if i+1 not in df.No.values:
            with open(csv_fname, "a", newline='') as csvfile: 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([i+1, count, too_close_pair_idx])
    
    # Move atoms apart
    new_cart_coords = struc.cart_coords
    for pair in too_close_pair_idx:
        pos0 = new_cart_coords[pair[0]]
        pos1 = new_cart_coords[pair[1]]
        dist_vector = pos1 - pos0
        dist_norm = np.linalg.norm(dist_vector)
        target_norm = min_dist + 0.1   # push further than min_dist to help pass the check the next time 
        new_pos1 = pos0 + dist_vector*(target_norm/dist_norm)  
        new_cart_coords[pair[1]] = new_pos1
    lat = struc.lattice
    atom_types = struc.species
    new_struc = Structure(lat, atom_types, new_cart_coords, coords_are_cartesian=True)
    
    return new_struc, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='gen.extxyz', help='path to generated structures (extxyz file)')
    parser.add_argument('--max_iter', type=int, default=100, help='maximum number of iterations allowed for moving atoms apart')
    parser.add_argument('--min_dist', type=float, default=1.5, help='minimum separation distance of atoms (unit: Angstrom)')
    parser.add_argument('--write_fname', type=str, default='gen_clean.extxyz', help='filename to write output extxyz file')
    
    args = parser.parse_args()
    
    # Load data
    data = read(args.data_path, index=':', format='extxyz')
    n_structures = len(data)
    
    ase_adaptor = AseAtomsAdaptor()
    processed_strucs = []   # Stores the processed structures
    iterations_list = []   # Stores the number of iterations for each structure
    csv_fname = 'too_close_count.csv'   # Name of csv that logs the number of unique pairs of atoms too close together
    
    for i in tqdm(range(n_structures)):
        # Construct Structure object
        struc = ase_adaptor.get_structure(data[i])
        
        # Move atoms apart if needed
        iterations = 0   # Initialize number of iterations of move_atoms_apart()
        count = 999   # Intialize count of atoms too close together to start check for atoms too close together
        while count > 0 and iterations < args.max_iter:
            # Keep moving atoms apart until bond distances are larger than min_dist
            struc, count = move_atoms_apart(struc, args.min_dist, i, csv_fname)
            iterations += 1    
    
        if count == 0:
            processed_strucs.append(ase_adaptor.get_atoms(struc))
            iterations_list.append(iterations)
        else:
            print('Failed to move atoms apart for structure index {}! Consider increasing max_iter.'.format(i))
            # Delete entry from csv file
            df_del = pd.read_csv(csv_fname, skipfooter=1)
            df_del.to_csv(csv_fname, index=False)    

    # Add number of iterations to csv
    df = pd.read_csv(csv_fname)
    df.insert(1, "iterations", iterations_list)
    df.to_csv(csv_fname, index=False)
    
    # Save processed structures
    print('Processed {}/{} structures successfully'.format(len(processed_strucs), n_structures))
    write(args.write_fname, processed_strucs, 'extxyz')
    

if  __name__ == '__main__':
    main()

