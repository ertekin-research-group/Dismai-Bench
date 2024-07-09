import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
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
    parser.add_argument('--data_gen_dir', type=str, default='.', help='path to directory containing cluster probability of generated structures (csv file)')
    parser.add_argument('--data_train_dir', type=str, default='/path/to/data/dismai_bench_train_ref_data/alloy_300K_narrow', help='path to directory containing cluster probability of training structures (csv file)')
    parser.add_argument('--n_strucs_boot', type=int, default=-1, help='number of structures in each bootstrap sample (if < 0, set to be equal to number of structures in gen dataset')
    parser.add_argument('--n_reps', type=int, default=1000, help='number of bootstrap repetitions')
    parser.add_argument('--conf_level', type=float, default=0.95, help='confidence level for confidence interval')
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed for bootstrapping')
    parser.add_argument('--n_strucs_ori', type=int, default=1000, help='number of structures originally generated using the generative model (to calculate the percentage of rejected structures)')
    
    args = parser.parse_args()
    df_prob_train = pd.read_csv(args.data_train_dir+'/train_cluster_prob.csv', index_col=0)
    df_prob_gen = pd.read_csv(args.data_gen_dir+'/cluster_prob.csv', index_col=0)
    try:
        df_vac = pd.read_csv(args.data_gen_dir+'/../vac_count.csv', index_col=0)
    except:
        df_vac = pd.read_csv(args.data_gen_dir+'/vac_count.csv', index_col=0)
    
    # Calculate percentage of rejected structures
    n_strucs_gen = len(df_prob_gen)   # Number of structures in gen dataset
    percent_rejected_strucs = 100.0 - (n_strucs_gen / args.n_strucs_ori * 100)
    assert percent_rejected_strucs <= 100.0, 'Percentage of rejected structures calculated to be > 100 %, please check that n_strucs_ori is set correctly'
    
    # Perform bootstrapping for cluster probability of good structures
    print('Bootstrapping the cluster probability distribution Euclidean distance...')
    rng = np.random.default_rng(args.seed)
    n_strucs_gen = len(df_prob_gen)   # Number of structures in gen dataset (only good structures)
    if args.n_strucs_boot < 0:
        n_strucs_boot = n_strucs_gen
    else:
        n_strucs_boot = args.n_strucs_boot
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    eu_dist_boot_prob_all = []   # Stores the euclidean distances of each bootstrap sample
    avg_prob_train = df_prob_train.mean()
    avg_prob_gen = df_prob_gen.mean()
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        # Get bootstrap sample
        df_boot = df_prob_gen.iloc[boot_struc_indices]
        # Get average cluster probability distribution
        avg_prob_boot = df_boot.mean()
        # Get Euclidean distance
        eu_dist_boot_prob = euclidean(avg_prob_train, avg_prob_boot)
        eu_dist_boot_prob_all.append(eu_dist_boot_prob)
    
    # Perform bootstrapping for percentage of good structures with vacancies
    print('Bootstrapping the percentage of accepted structures with vacancies...')
    rng = np.random.default_rng(args.seed)
    n_strucs_gen = len(df_vac)   # Number of structures in gen dataset (only good structures)
    if args.n_strucs_boot < 0:
        n_strucs_boot = n_strucs_gen
    else:
        n_strucs_boot = args.n_strucs_boot
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    vac_percent_boot_all = []   # Stores the average percentage of vacancies of each bootstrap sample
    vac_percent_gen = (df_vac.vac_count > 0).mean() * 100
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        # Get bootstrap sample
        df_boot = df_vac.iloc[boot_struc_indices]
        # Get percentage of good structures with vacancies
        vac_percent_boot = (df_boot.vac_count > 0).mean() * 100
        vac_percent_boot_all.append(vac_percent_boot)
    
    # Calculate bootstrap confidence intervals
    eu_dist_prob = euclidean(avg_prob_train, avg_prob_gen)
    eu_dist_prob_error_low, eu_dist_prob_error_high = get_conf_interval(eu_dist_prob,  
                                                                        eu_dist_boot_prob_all, 
                                                                        args.conf_level)
    
    vac_percent_error_low, vac_percent_error_high = get_conf_interval(vac_percent_gen,  
                                                                      vac_percent_boot_all, 
                                                                      args.conf_level)
    
    # Save results
    df_results = pd.DataFrame.from_dict({'metric': ['Rejected structures (%)',
                                                    'Cluster probability Euclidean distance', 
                                                    'Accepted structures with vacancies (%)'],
                                         'value': [percent_rejected_strucs,
                                                   eu_dist_prob,
                                                   vac_percent_gen],
                                         'lower error': [None,
                                                         eu_dist_prob_error_low,
                                                         vac_percent_error_low],
                                         'upper error': [None,
                                                         eu_dist_prob_error_high,
                                                         vac_percent_error_high]
                                         }, 
                                        orient='columns')
    df_results.to_csv('alloy_metrics.csv', index=False)


if  __name__ == '__main__':
    main()
