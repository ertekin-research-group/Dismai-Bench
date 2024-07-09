from ase.io import read, write
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
import argparse
from tqdm import tqdm
from distutils.util import strtobool
import pandas as pd
from pymatgen.core.sites import PeriodicSite
import os
import numpy as np

def get_rotated_basis_vectors(struc, struc_idx, margin=0.4):
    """
    Gets the basis vectors in the orientation of a rotated FCC structure

    Parameters
    ----------
    struc : Pymatgen Structure object
    struc_idx: Index number of structure
    margin : Bond distance margin of error (unit: Angstrom) for determining 2nd nearest neighbors

    Returns
    -------
    basis_rot: 3x3 NumPy array of basis vectors
    atom_idx: index of center atom used to obtain basis vectors

    """
    done = False
    atom_idx = -1   # Initialize center atom index for calculating basis vectors
    # Iterate over all atoms as the center atom until basis vectors are obtained 
    while not done:
        atom_idx += 1
        if atom_idx >= len(struc):
            atom_idx = -1
            print(f'Basis vectors cannot be determined for Structure index {struc_idx}')
            return None, atom_idx
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
                    c_new /= np.linalg.norm(c_new)
                    basis_rot = np.array([a_new, b_new, c_new])
                    done = True
                    break
        
    return basis_rot, atom_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='gen.extxyz', help='path to generated structures (.extxyz file)')
    parser.add_argument('--ref_struc', type=str, default='alloy_ref_struc.extxyz', help='path to reference structure of perfect crystal (.extxyz file)')
    parser.add_argument('--margin', type=float, default=0.8, help='margin of error for distance between designated cluster coordinate and actual site coordinate (unit: Angstrom)')
    parser.add_argument('--rotate_ref_lat', default=True, type=lambda x:bool(strtobool(x)), help='determines if reference structure is rotated to match generated structure lattice')
    parser.add_argument('--write_fname', type=str, default='gen_clean.extxyz', help='filename to write output extxyz file')
    
    args = parser.parse_args()
    
    if not os.path.exists('cluster_counts'):
        os.makedirs('cluster_counts')
    
    ase_adaptor = AseAtomsAdaptor()
    struc_all = read(args.data, index=':', format='extxyz')
    struc_ref_ori = read(args.ref_struc, index=0, format='extxyz')
    if args.rotate_ref_lat:
        # Create a 3x3x3 supercell of the reference structure
        struc_ref_ori_333 = make_supercell(struc_ref_ori, 
                                           [[3, 0, 0], 
                                            [0, 3, 0],
                                            [0, 0, 3]]
                                           )
        frac_coords_ref_333 = struc_ref_ori_333.get_scaled_positions()
        # Shift the coordinates to be centered around the origin
        frac_coords_ref_333 -= 0.5
    struc_ref_ori = ase_adaptor.get_structure(struc_ref_ori)
    n_atoms_ori = len(struc_ref_ori)
    
    # Create template structure with no atoms (for building cleaned-up structures)
    struc_template = struc_ref_ori.copy()
    struc_template.remove_sites(list(range(len(struc_template))))
    
    struc_clean_all = []   # Stores all cleaned-up structures
    n_atoms_removed_all = []   # Stores the number of removed atoms for all accepted structures
    accepted_indexes = []   # Stores the indexes of accepted structures
    for i in tqdm(range(len(struc_all))):
        struc = ase_adaptor.get_structure(struc_all[i])
        struc_clean = struc_template.copy()
        struc_ref = struc_ref_ori.copy()
        check_indexes = []   # Stores atom indexes that should be manually checked if on the lattice
        
        if args.rotate_ref_lat:
            # Rotate the 3x3x3 reference coordinates
            basis_rot, atom_idx = get_rotated_basis_vectors(struc, i)
            if atom_idx < 0:
                # Skip structure
                continue
            frac_coords_ref = np.matmul(frac_coords_ref_333, basis_rot)
            # Keep only the coordinates of a 1x1x1 cell centered around the origin
            frac_coords_ref = frac_coords_ref[((frac_coords_ref[:,0] >  -1/6) &
                                               (frac_coords_ref[:,0] <=  1/6) &
                                               (frac_coords_ref[:,1] >  -1/6) &
                                               (frac_coords_ref[:,1] <=  1/6) &
                                               (frac_coords_ref[:,2] >  -1/6) &
                                               (frac_coords_ref[:,2] <=  1/6)
                                               )]
            # Rescale and shift the coordinates back into a 1x1x1 cell (centered around [0.5, 0.5, 0.5])
            frac_coords_ref *= 3
            frac_coords_ref += 0.5
            # Construct reference structure
            struc_ref = struc_template.copy()
            for j in range(len(frac_coords_ref)):
                struc_ref.append('H', frac_coords_ref[j], coords_are_cartesian=False)
            
            # Translate reference lattice to match generated structure lattice
            pin_site = PeriodicSite('H', struc[atom_idx].frac_coords, struc_ref_ori.lattice)
            margin_translate = 5
            _ , struc_site_idx_all, _ , dist_all = struc_ref.get_neighbor_list(margin_translate, [pin_site])
            if len(struc_site_idx_all) > 0:
                # Get generated atom closest to pin_site
                struc_site_idx = struc_site_idx_all[dist_all.argmin()]
                struc_ref = ase_adaptor.get_atoms(struc_ref)
                frac_coords = struc_ref.get_scaled_positions()
                frac_coords += pin_site.frac_coords - frac_coords[struc_site_idx]
                struc_ref.set_scaled_positions(frac_coords)
                struc_ref.wrap()
                struc_ref = ase_adaptor.get_structure(struc_ref)
            else:
                raise Exception('Atom not found near pin_site to translate the reference lattice, please increase margin_translate')   
        
        # Construct clean structures
        for j in range(len(struc)): 
            site = struc[j]
            # Check if atom is on lattice
            _ , struc_site_idx_all, _ , dist_all = struc_ref.get_neighbor_list(args.margin, [site])
            if len(struc_site_idx_all) > 0:
                if len(struc_clean) == 0:
                    struc_clean.append(site.specie,
                                       site.frac_coords,
                                       coords_are_cartesian=False)
                else:
                    # Check if atom is on top of another atom
                    _ , struc_site_idx_all, _ , dist_all = struc_clean.get_neighbor_list(args.margin, [site])
                    if len(struc_site_idx_all) == 0:
                        struc_clean.append(site.specie,
                                           site.frac_coords,
                                           coords_are_cartesian=False)
            else:
                check_indexes.append(j)
        
        if len(check_indexes) > 0:
            # Atoms not on reference structure lattice can still be on lattice (happens when the lattice spacing is smaller than the reference structure, so there are more atoms than expected)
            # Check if atom is on lattice manually, accept if a 2nd nearest neighbor is found
            nb_dir_all = [basis_rot[0], -basis_rot[0],
                          basis_rot[1], -basis_rot[1],
                          basis_rot[2], -basis_rot[2]]   # Directions of 2nd nearest neighbor
            n_manually_added = 999   # Intialize count for while loop
            while n_manually_added > 0:
                # As long as atoms have been added to the structure, keep running the loop
                n_manually_added = 0
                for j in range(len(check_indexes)-1, -1 , -1):
                    site = struc[check_indexes[j]]
                    # Check if the atom is on top of another atom
                    _ , struc_site_idx_all, _ , dist_all = struc_clean.get_neighbor_list(args.margin, [site])
                    if len(struc_site_idx_all) == 0:
                        # Check all 2nd nearest neighbor sites if any neighbor is present
                        for nb_dir in nb_dir_all:
                            check_site = PeriodicSite('H', site.frac_coords + nb_dir*0.25, struc_clean.lattice)
                            _ , struc_site_idx_all, _ , dist_all = struc_clean.get_neighbor_list(args.margin, [check_site])
                            if len(struc_site_idx_all) > 0:
                                # 2nd nearest neighbor found, accept the atom
                                struc_clean.append(site.specie,
                                                   site.frac_coords,
                                                   coords_are_cartesian=False)
                                n_manually_added += 1
                                check_indexes.pop(j)
                                break
                    else:
                        # Atom is on top of another atom
                        check_indexes.pop(j)
        
        n_atoms_removed = n_atoms_ori - len(struc_clean)
        if n_atoms_removed > 50:
            # Too many atoms not on lattice, skip structure
            print(f'Structure index {i} skipped, {n_atoms_removed} atoms not on lattice')
            continue
        struc_clean = struc_clean.get_sorted_structure()
        struc_clean = ase_adaptor.get_atoms(struc_clean)
        struc_clean_all.append(struc_clean)
        n_atoms_removed_all.append(n_atoms_removed)
        accepted_indexes.append(i)
    
    write('cluster_counts/' + args.write_fname, struc_clean_all, format='extxyz')
    df = pd.DataFrame.from_dict({'structure_index': accepted_indexes,
                                 'number_atoms_removed': n_atoms_removed_all},
                                orient='columns')
    df.to_csv('atoms_removed_count.csv', index=False)
    print(f'{len(struc_clean_all)}/{len(struc_all)} structures accepted.')
    
    
if  __name__ == '__main__':
    main()

