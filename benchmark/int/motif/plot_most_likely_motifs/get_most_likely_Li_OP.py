import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


def get_conf_interval(stat, boot_stats, conf_level):
    """
    Calculates the bootstrap pivotal confidence interval.
    Follows equation 8.6 of https://doi.org/10.1007/978-0-387-21736-9_8
    
    Parameters
    ----------
    stat : statistic calculated using original data
    boot_stats : list of statistics calculated from bootstrap samples
    conf_level : confidence level expressed as a fraction (not percentage)

    Returns
    -------
    error_low: size of lower error
    error_high: size of upper error
    
    """
    low_percentile = np.percentile(boot_stats, (1-conf_level)/2*100)
    high_percentile = np.percentile(boot_stats, (conf_level + (1-conf_level)/2)*100)
    error_low = high_percentile - stat
    error_high = stat - low_percentile
    
    return error_low, error_high

def get_n_Cl_distribution(df, op_pct, max_cn):
    """
    Subdivide each order parameter based on number of Cl neighbors

    Parameters
    ----------
    df : pandas.DataFrame of all fingerprints 
    op_pct : pandas.Series of the most likely order parameter percentage distribution
    max_cn : maximum coordination number of all order parameters in op_pct

    Returns
    -------
    n_Cl_pct: dictionary that stores, for each order parameter, the percentage of 
        the order parameter with number of Cl neighbors from 0 to max_cn

    """
    n_Cl_pct = {}
    for op in op_pct.index:
        n_Cl_pct[op] = {}
        df_op = df.loc[df['most likely OP'] == op]
        series_n_Cl_freq = df_op['Cl count (most likely CN)'].value_counts(sort=False)
        series_n_Cl_pct = series_n_Cl_freq / len(df) * 100
        series_n_Cl_pct.index = series_n_Cl_pct.index.map(int)
        for n_Cl in range(0, max_cn+1):
            if n_Cl in series_n_Cl_pct.index:
                n_Cl_pct[op][str(n_Cl)+' Cl'] = series_n_Cl_pct[n_Cl]
            else:
                n_Cl_pct[op][str(n_Cl)+' Cl'] = 0
    
    return n_Cl_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_gen', type=str, default='../cnn_stats_Li.csv', help='path to coordination motif csv file of generated structures')
    parser.add_argument('--data_train', type=str, default='/path/to/data/dismai_bench_train_ref_data/int/train_motif/cnn_stats_Li.csv', help='path to coordination motif csv file of training structures')
    parser.add_argument('--op_pct_threshold', type=float, default=1.0, help='only order parameters with occurrence >= this percentage threshold in the train dataset are plotted ')
    parser.add_argument('--n_strucs_boot', type=int, default=-1, help='number of structures in each bootstrap sample (if < 0, set to be equal to number of structures in gen dataset')
    parser.add_argument('--n_reps', type=int, default=1000, help='number of bootstrap repetitions')
    parser.add_argument('--conf_level', type=float, default=0.95, help='confidence level for confidence interval')
    parser.add_argument('--n_atoms', type=int, default=72, help='number of atoms of this atom type in each structure')
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed for bootstrapping')
    
    args = parser.parse_args()
    
    # Train dataset: get distribution of most likely order parameter
    df_train = pd.read_csv(args.data_train)
    op_freq_train = df_train['most likely OP'].value_counts(sort=False)
    sort_order = {'cuboctahedral CN_12': 0,
                  'hexagonal bipyramidal CN_8':1, 'body-centered cubic CN_8':2,
                  'pentagonal bipyramidal CN_7':3, 'hexagonal pyramidal CN_7':4,
                  'pentagonal pyramidal CN_6':5, 'octahedral CN_6':6, 'hexagonal planar CN_6':7,
                  'trigonal bipyramidal CN_5':8, 'square pyramidal CN_5':9, 'pentagonal planar CN_5':10,
                  'trigonal pyramidal CN_4':11, 'see-saw-like CN_4':12, 'rectangular see-saw-like CN_4':13, 'tetrahedral CN_4':14, 'square co-planar CN_4':15,
                  'T-shaped CN_3':16,'trigonal non-coplanar CN_3':17, 'trigonal planar CN_3':18,
                  'linear CN_2':19, 'bent 150 degrees CN_2':20, 'bent 120 degrees CN_2':21, 'water-like CN_2':22, 'L-shaped CN_2':23,
                  'sgl_bd CN_1':24}
    op_freq_train = op_freq_train.sort_index(key=lambda x: x.map(sort_order))
    op_pct_train = op_freq_train / op_freq_train.sum() * 100
    # Filter to include only order parameters above threshold
    op_pct_train = op_pct_train[op_pct_train >= args.op_pct_threshold]
    # Subdivide each order parameter based on number of Cl neighbors
    max_cn = int(op_pct_train.index[0].split('_')[-1])   # Maximum coordination number in filtered order parameters
    n_Cl_pct_train = get_n_Cl_distribution(df_train, op_pct_train, max_cn)
    
    # Gen dataset: get distribution of most likely order parameter
    df_gen = pd.read_csv(args.data_gen)
    op_freq_gen = df_gen['most likely OP'].value_counts(sort=False)
    op_freq_gen = op_freq_gen.sort_index(key=lambda x: x.map(sort_order))
    op_pct_gen = op_freq_gen / op_freq_gen.sum() * 100
    # Filter to include only order parameters above threshold in train dataset
    # (order parameters in train dataset but not in gen dataset are set to zero)
    op_pct_gen_filter = pd.Series(0, index=op_pct_train.index)
    for op in op_pct_gen_filter.index:
        if op in op_pct_gen.index:
            op_pct_gen_filter[op] = op_pct_gen[op]
    op_pct_gen = op_pct_gen_filter
    # Subdivide each order parameter based on number of Cl neighbors
    n_Cl_pct_gen = get_n_Cl_distribution(df_gen, op_pct_gen, max_cn)
    
    # Perform bootstrapping
    op_pct_boot_all = {}   # Stores the order parameter percentage of each bootstrap sample 
    for op in op_pct_train.index:
        op_pct_boot_all[op] = []
    n_Cl_pct_boot_all = {}   # Stores the n_Cl distribution of each bootstrap sample 
    for op in op_pct_train.index:
        n_Cl_pct_boot_all[op] = {}
        for n_Cl in range(0, max_cn+1):
            n_Cl_pct_boot_all[op][str(n_Cl)+' Cl'] = []
    rng = np.random.default_rng(args.seed)
    n_strucs_gen = df_gen['structure_number'].iloc[-1]   # Number of structures in gen dataset
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    if args.n_strucs_boot < 0:
        n_strucs_boot = n_strucs_gen
    else:
        n_strucs_boot = args.n_strucs_boot
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        boot_atom_indices = []
        for i in boot_struc_indices:
            boot_atom_indices += list(range(i*args.n_atoms, (i+1)*args.n_atoms))
        # Get bootstrap sample
        df_boot = df_gen.iloc[boot_atom_indices]
        # Get distribution of most likely order parameter
        op_freq_boot = df_boot['most likely OP'].value_counts(sort=False)
        op_pct_boot = op_freq_boot / op_freq_boot.sum() * 100
        # Filter to include only order parameters above threshold in train dataset
        # (order parameters in train dataset but not in bootstrap sample are set to zero)
        op_pct_boot_filter = pd.Series(0, index=op_pct_train.index)
        for op in op_pct_boot_all.keys():
            if op in op_pct_boot.index:
                op_pct_boot_filter[op] = op_pct_boot[op]
                op_pct_boot_all[op].append(op_pct_boot[op])
            else:
                op_pct_boot_all[op].append(0)
        op_pct_boot = op_pct_boot_filter
        # Subdivide each order parameter based on number of Cl neighbors
        n_Cl_pct_boot = get_n_Cl_distribution(df_boot, op_pct_boot, max_cn)
        for op in n_Cl_pct_boot_all.keys():
            for n_Cl in range(0, max_cn+1):
                n_Cl_pct_boot_all[op][str(n_Cl)+' Cl'].append(n_Cl_pct_boot[op][str(n_Cl)+' Cl'])    
    
    # Calculate bootstrap confidence interval of op_pct
    op_pct_gen_error_low = {}
    op_pct_gen_error_high = {}
    for op in op_pct_gen.index:
        error_low, error_high = get_conf_interval(op_pct_gen[op],  
                                                  op_pct_boot_all[op], 
                                                  args.conf_level)
        op_pct_gen_error_low[op] = error_low
        op_pct_gen_error_high[op] = error_high
    
    # Calculate bootstrap confidence interval of n_Cl_pct
    n_Cl_pct_gen_error_low = {}
    n_Cl_pct_gen_error_high = {}
    for op in op_pct_train.index:
        n_Cl_pct_gen_error_low[op] = {}
        n_Cl_pct_gen_error_high[op] = {}
        for n_Cl in range(0, max_cn+1):
            error_low, error_high = get_conf_interval(n_Cl_pct_gen[op][str(n_Cl)+' Cl'],  
                                                      n_Cl_pct_boot_all[op][str(n_Cl)+' Cl'], 
                                                      args.conf_level)
            n_Cl_pct_gen_error_low[op][str(n_Cl)+' Cl lower error'] = error_low
            n_Cl_pct_gen_error_high[op][str(n_Cl)+' Cl upper error'] = error_high
    
    # Save results to csv file
    df_results = pd.concat([op_pct_train, op_pct_gen], axis=1, 
                           keys=['Train set (%)', 'Gen set (%)'])
    df_results['Gen set lower error (%)'] = df_results.index.map(op_pct_gen_error_low)
    df_results['Gen set upper error (%)'] = df_results.index.map(op_pct_gen_error_high)
    df_n_Cl_train = pd.DataFrame.from_dict(n_Cl_pct_train, orient='index')
    df_n_Cl_train.columns = 'Train set ' + df_n_Cl_train.columns + ' (%)'
    df_n_Cl_gen = pd.DataFrame.from_dict(n_Cl_pct_gen, orient='index')
    df_n_Cl_gen.columns = 'Gen set ' + df_n_Cl_gen.columns + ' (%)'
    df_n_Cl_gen_error_low = pd.DataFrame.from_dict(n_Cl_pct_gen_error_low, orient='index')
    df_n_Cl_gen_error_low.columns = 'Gen set ' + df_n_Cl_gen_error_low.columns + ' (%)'
    df_n_Cl_gen_error_high = pd.DataFrame.from_dict(n_Cl_pct_gen_error_high, orient='index')
    df_n_Cl_gen_error_high.columns = 'Gen set ' + df_n_Cl_gen_error_high.columns + ' (%)'
    df_results = pd.concat([df_results,
                            df_n_Cl_train,
                            df_n_Cl_gen,
                            df_n_Cl_gen_error_low,
                            df_n_Cl_gen_error_high
                            ], 
                           axis=1)
    df_results.to_csv('most_likely_Li_OP.csv', index=True)


if  __name__ == '__main__':
    main()
