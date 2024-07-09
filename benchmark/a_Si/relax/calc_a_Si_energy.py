from ase.io import read
import csv
import argparse
from quippy.potential import Potential
from distutils.util import strtobool
from math import ceil

import warnings
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split relaxation into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_clean.extxyz', help='path to data (extxyz file)')
    parser.add_argument('--model_path', type=str, default='/path/to/data/potentials/GAP_PRX_Si/gp_iter6_sparse9k.xml', help='path to GAP model')
    
    args = parser.parse_args()
    batch = args.batch
    
    if args.split_batches:
        data = read(filename='../'+args.data_path, index=':', format='extxyz')
    else:
        data = read(filename=args.data_path, index=':', format='extxyz')
    csv_fname = 'energy_results.csv'   # For saving energy results
    
    # Get starting and ending indexes of structures to calculate
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
    
    # Write csv header
    with open(csv_fname, "a", newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['batch', 'structure_number', 'E (eV/atom)'])
    
    # Initialize GAP potential
    Si_potential = Potential(init_args='Potential xml_label="GAP_2017_6_17_60_4_3_56_165"',
                             param_filename=args.model_path)
    
    for i in range(i_start, i_end+1):
        print("Structure number", i+1)
        struc = data[i].copy()
        struc.set_calculator(Si_potential)
        energy_per_atom = struc.get_potential_energy() / len(struc)
        
        with open(csv_fname, "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([batch, i+1, energy_per_atom])


if  __name__ == '__main__':
    main()
