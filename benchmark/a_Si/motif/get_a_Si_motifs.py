from matminer.featurizers.structure.sites import CrystalNNFingerprint
from tqdm import tqdm
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import csv
import argparse
from distutils.util import strtobool
from math import ceil


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
    csv_fname = 'cnn_stats_Si.csv'   # For saving results
    
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
    
    # Define order parameters to analyze
    op_types_dict = {1: ['wt', 'sgl_bd'],
                     2: ['wt',
                      'L-shaped',
                      'water-like',
                      'bent 120 degrees',
                      'bent 150 degrees',
                      'linear'],
                     3: ['wt', 'trigonal planar', 'trigonal non-coplanar', 'T-shaped'],
                     4: ['wt',
                      'square co-planar',
                      'tetrahedral',
                      'rectangular see-saw-like',
                      'see-saw-like',
                      'trigonal pyramidal'],
                     5: ['wt', 'pentagonal planar', 'square pyramidal', 'trigonal bipyramidal'],
                     6: ['wt', 'hexagonal planar', 'octahedral', 'pentagonal pyramidal'],
                     7: ['wt', 'hexagonal pyramidal', 'pentagonal bipyramidal'],
                     8: ['wt', 'body-centered cubic', 'hexagonal bipyramidal'],
                     9: ['wt', 'q2', 'q4', 'q6'],
                     10: ['wt', 'q2', 'q4', 'q6'],
                     11: ['wt', 'q2', 'q4', 'q6'],
                     12: ['wt', 'cuboctahedral', 'q2', 'q4', 'q6'],
                     13: ['wt'],
                     14: ['wt'],
                     15: ['wt'],
                     16: ['wt'],
                     17: ['wt'],
                     18: ['wt'],
                     19: ['wt'],
                     20: ['wt'],
                     21: ['wt'],
                     22: ['wt'],
                     23: ['wt'],
                     24: ['wt']
                     }
    cnn = CrystalNNFingerprint(op_types=op_types_dict, cation_anion=False)
    
    # Write csv header
    with open(csv_fname, "w", newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['structure_number', 'Si_number'] + cnn.feature_labels())
    
    # Analyze coordination environment
    print('Analyzing coordination environment')
    ase_adaptor = AseAtomsAdaptor()
    for i in tqdm(range(i_start, i_end+1)):
        atoms = data[i]
        struc = ase_adaptor.get_structure(atoms)
        
        indices_Si = [idx for idx, site in enumerate(struc) if site.species_string == 'Si']
        for j in indices_Si:
            stat = cnn.featurize(struc, j)
            with open(csv_fname, "a", newline='') as csvfile: 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([i+1, j+1] + stat)    
    
    ## Get the most likely coordination number and order parameter
    print('Processing output file')
    df = pd.read_csv(csv_fname)
    
    # Find the most likely coordination number
    df_cn = df.filter(like='wt CN')
    df['most likely CN'] = df_cn.idxmax(axis=1).str.split('_').str[-1].astype(int)
    df['CN likelihood'] = df_cn.max(axis=1)
    
    # Find the most likely order parameter
    OP_type_all = []
    OP_val_all = []
    for i in tqdm(range(len(df))):
        cn = df['most likely CN'][i]
        df_op = df.iloc[i].filter(like='CN_'+str(cn))
        df_op = df_op.drop(labels=['wt CN_'+str(cn)])
        if len(df_op) > 0:
            OP_type_all.append(df_op.idxmax(axis=0))
            OP_val_all.append(df_op.max(axis=0))
        else:
            OP_type_all.append('None CN_'+str(cn))
            OP_val_all.append(None)
    df['most likely OP'] = pd.Series(OP_type_all)
    df['OP value'] = pd.Series(OP_val_all)
    
    # Rearrange new columns to the front
    cols = list(df.columns)
    cols = cols[:2] + cols[-4:] + cols[2:-4]
    df = df[cols]
    df.to_csv(csv_fname, index=False)


if  __name__ == '__main__':
    main()
