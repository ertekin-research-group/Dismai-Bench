import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_gen', type=str, default='cnn_stats_Si.csv', help='path to coordination motif csv file of generated structures')
    parser.add_argument('--data_train', type=str, default='/path/to/data/dismai_bench_train_ref_data/a_Si/train_motif/cnn_stats_Si_train.csv', help='path to coordination motif csv file of training structures')
    parser.add_argument('--n_strucs_boot', type=int, default=-1, help='number of structures in each bootstrap sample (if < 0, set to be equal to number of structures in gen dataset')
    parser.add_argument('--n_reps', type=int, default=1000, help='number of bootstrap repetitions')
    parser.add_argument('--conf_level', type=float, default=0.95, help='confidence level for confidence interval')
    parser.add_argument('--n_atoms', type=int, default=256, help='number of atoms in each structure')
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed for bootstrapping')
    parser.add_argument('--n_strucs_ori', type=int, default=1000, help='number of structures originally generated using the generative model (to calculate the percentage of failed structures)')
    
    args = parser.parse_args()
    df_train = pd.read_csv(args.data_train)
    df_gen = pd.read_csv(args.data_gen)
    
    # Get average fingerprint of train dataset
    fingerprint_avg_train = df_train.iloc[:, 6:67].mean().to_numpy()
    
    # Get fingerprint of each structure in gen dataset (used for bootstrapping)
    fingerprint_gen_all = []
    n_strucs_gen = df_gen['structure_number'].iloc[-1]   # Number of structures in gen dataset
    for i in range(n_strucs_gen):
        atom_indices = list(range(i*args.n_atoms, (i+1)*args.n_atoms))
        df_gen_struc = df_gen.iloc[atom_indices]
        # Average fingerprints of all atoms in the structure
        fingerprint = df_gen_struc.iloc[:, 6:67].mean().to_numpy()
        fingerprint_gen_all.append(fingerprint)
    fingerprint_gen_all = np.array(fingerprint_gen_all)
    # Get average fingerprint of gen dataset
    fingerprint_avg_gen = np.mean(fingerprint_gen_all, axis=0)
    
    # Perform bootstrapping
    rng = np.random.default_rng(args.seed)
    if args.n_strucs_boot < 0:
        n_strucs_boot = n_strucs_gen
    else:
        n_strucs_boot = args.n_strucs_boot
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    eu_dist_boot_all = []   # Stores the euclidean distances of each bootstrap sample 
    cos_dist_boot_all = []   # Stores the cosine distance of each bootstrap sample 
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        # Get fingerprints of all structures in bootstrap sample
        fingerprint_boot_all = fingerprint_gen_all[boot_struc_indices]
        # Average the fingerprints in the bootstrap sample
        fingerprint_avg_boot = np.mean(fingerprint_boot_all, axis=0)
        eu_dist_boot = euclidean(fingerprint_avg_train, fingerprint_avg_boot)
        cos_dist_boot = cosine(fingerprint_avg_train, fingerprint_avg_boot)
        eu_dist_boot_all.append(eu_dist_boot)
        cos_dist_boot_all.append(cos_dist_boot)
    
    # Calculate bootstrap confidence interval
    eu_dist = euclidean(fingerprint_avg_train, fingerprint_avg_gen)
    cos_dist = cosine(fingerprint_avg_train, fingerprint_avg_gen)
    eu_dist_error_low, eu_dist_error_high = get_conf_interval(eu_dist, eu_dist_boot_all, args.conf_level)
    cos_dist_error_low, cos_dist_error_high = get_conf_interval(cos_dist, cos_dist_boot_all, args.conf_level)
    
    # Calculate percentage of failed structures
    percent_failed_strucs = 100.0 - (n_strucs_gen / args.n_strucs_ori * 100)
    assert percent_failed_strucs <= 100.0, 'Percentage of failed structures calculated to be > 100 %, please check that n_strucs_ori is set correctly'
    
    # Save results
    df_results = pd.DataFrame.from_dict({'metric': ['Euclidean distance', 'Cosine distance', 'Failed structures (%)'],
                                         'value': [eu_dist, cos_dist, percent_failed_strucs],
                                         'lower error': [eu_dist_error_low, cos_dist_error_low, None],
                                         'upper error': [eu_dist_error_high, cos_dist_error_high, None]}, 
                                        orient='columns')
    df_results.to_csv('a_Si_motif_metrics.csv', index=False)


if  __name__ == '__main__':
    main()
