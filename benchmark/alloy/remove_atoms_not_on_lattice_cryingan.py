from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
import argparse
from tqdm import tqdm
from distutils.util import strtobool
import pandas as pd
from pymatgen.core.sites import PeriodicSite
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='gen.extxyz', help='path to generated structures (.extxyz file)')
    parser.add_argument('--ref_struc', type=str, default='alloy_ref_struc.extxyz', help='path to reference structure of perfect crystal (.extxyz file)')
    parser.add_argument('--margin', type=float, default=0.8, help='margin of error for distance between designated cluster coordinate and actual site coordinate (unit: Angstrom)')
    parser.add_argument('--translate_ref_lat', default=False, type=lambda x:bool(strtobool(x)), help='determines if reference structure is translated to match generated structure lattice')
    parser.add_argument('--write_fname', type=str, default='gen_clean.extxyz', help='filename to write output extxyz file')
    
    args = parser.parse_args()
    
    if not os.path.exists('cluster_counts'):
        os.makedirs('cluster_counts')
    
    ase_adaptor = AseAtomsAdaptor()
    struc_all = read(args.data, index=':', format='extxyz')
    struc_ref_ori = read(args.ref_struc, index=0, format='extxyz')
    struc_ref_ori = ase_adaptor.get_structure(struc_ref_ori)
    n_atoms_ori = len(struc_ref_ori)
    
    # Create template structure with no atoms (for building cleaned-up structures)
    struc_template = struc_ref_ori.copy()
    struc_template.remove_sites(list(range(len(struc_template))))
    
    struc_clean_all = []   # Stores all cleaned-up structures
    vac_count_all = []   # Stores the vacancy count of all structures
    for i in tqdm(range(len(struc_all))):
        struc = ase_adaptor.get_structure(struc_all[i])
        struc_clean = struc_template.copy()
        if args.translate_ref_lat:
            # Translate reference lattice to match generated structure lattice
            origin_site = PeriodicSite('H', [1e-06, 1e-06, 1e-06], struc_ref_ori.lattice)
            margin_translate = 5
            _ , struc_site_idx_all, _ , dist_all = struc.get_neighbor_list(margin_translate, [origin_site])
            if len(struc_site_idx_all) > 0:
                # Get generated atom closest to origin
                struc_site_idx = struc_site_idx_all[dist_all.argmin()]
                struc_ref = ase_adaptor.get_atoms(struc_ref_ori)
                frac_coords = struc_ref.get_scaled_positions()
                frac_coords += struc[struc_site_idx].frac_coords - origin_site.frac_coords
                struc_ref.set_scaled_positions(frac_coords)
                struc_ref.wrap()
                struc_ref = ase_adaptor.get_structure(struc_ref)
            else:
                raise Exception('Atom not found near origin to translate the reference lattice, please increase margin_translate')
        else:
            struc_ref = struc_ref_ori
        for site in struc_ref:
            _ , struc_site_idx_all, _ , dist_all = struc.get_neighbor_list(args.margin, [site])
            if len(struc_site_idx_all) > 0:
                # Get generated atom closest to actual site coordinate
                struc_site_idx = struc_site_idx_all[dist_all.argmin()]
                struc_clean.append(struc[struc_site_idx].specie,
                                   site.frac_coords,
                                   coords_are_cartesian=False)
        struc_clean = struc_clean.get_sorted_structure()
        struc_clean = ase_adaptor.get_atoms(struc_clean)
        struc_clean_all.append(struc_clean)
        vac_count_all.append(n_atoms_ori - len(struc_clean))
    
    write('cluster_counts/' + args.write_fname, struc_clean_all, format='extxyz')
    df_vac = pd.DataFrame.from_dict({'structure_index': list(range(len(struc_all))),
                                     'vac_count': vac_count_all},
                                    orient='columns')
    df_vac.to_csv('vac_count.csv', index=False)
    print(f'Maximum vacancy count is {df_vac["vac_count"].max()}')
    
    
if  __name__ == '__main__':
    main()

