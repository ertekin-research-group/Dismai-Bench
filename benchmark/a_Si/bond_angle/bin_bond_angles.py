import pandas as pd
from ast import literal_eval
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='bond_angles.csv', help='path to bond angle data (csv file)')
    parser.add_argument('--bin_width', type=float, default=0.5, help='bin width in angle degrees')
    parser.add_argument('--min_bin_edge', type=float, default=0.0, help='minimum bin edge in angle degrees')
    parser.add_argument('--max_bin_edge', type=float, default=180.0, help='maximum bin edge in angle degrees')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_path, converters={'bond_angle_list': literal_eval})
    bin_edges = np.arange(args.min_bin_edge, args.max_bin_edge+args.bin_width, args.bin_width)
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Bin the bond angles for each structure and normalize to get its distribution
    bin_counts_all = {}   # Stores the bin counts for all structures
    for i in tqdm(range(len(df))):
        bond_angles_all = df['bond_angle_list'].iloc[i]
        assert min(bond_angles_all) >= args.min_bin_edge, f'Please increase min_bin_edge, bond angle of {round(min(bond_angles_all), 5)}° found'
        assert max(bond_angles_all) <= args.max_bin_edge, f'Please increase max_bin_edge, bond angle of {round(max(bond_angles_all), 5)}° found'
        
        #bin_counts = pd.cut(bond_angles_all, bin_edges, labels=False, include_lowest=True)
        bin_counts = pd.cut(pd.Series(bond_angles_all), bin_edges, include_lowest=True)
        bin_counts = bin_counts.value_counts(normalize=True, sort=False).to_numpy()
        bin_counts_all[i+1] = bin_counts
    
    # Save the bin counts
    df_dist = pd.DataFrame.from_dict(bin_counts_all, orient='index', columns=bin_midpoints)
    df_dist.index.name = 'structure_number'
    df_dist.to_csv('bond_angle_distribution.csv', index=True)


if  __name__ == '__main__':
    main()
