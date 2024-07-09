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
    parser.add_argument('--data_gen', type=str, default='bond_angle_distribution.csv', help='path to bond angle distribution of generated structures (csv file)')
    parser.add_argument('--data_train', type=str, default='/path/to/data/dismai_bench_train_ref_data/a_Si/train_bond_angle/bond_angle_distribution_train.csv', help='path to bond angle distribution of training structures (csv file)')
    parser.add_argument('--n_strucs_boot', type=int, default=-1, help='number of structures in each bootstrap sample (if < 0, set to be equal to number of structures in gen dataset')
    parser.add_argument('--n_reps', type=int, default=1000, help='number of bootstrap repetitions')
    parser.add_argument('--conf_level', type=float, default=0.95, help='confidence level for confidence interval')
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed for bootstrapping')
    
    args = parser.parse_args()
    df_train = pd.read_csv(args.data_train, index_col=0)
    df_gen = pd.read_csv(args.data_gen, index_col=0)
    
    # Get average bond angle distribution
    bd_train_avg = df_train.mean(axis=0).to_numpy()
    bd_gen_avg = df_gen.mean(axis=0).to_numpy()
    
    # Perform bootstrapping
    rng = np.random.default_rng(args.seed)
    n_strucs_gen = len(df_gen)   # Number of structures in gen dataset
    if args.n_strucs_boot < 0:
        n_strucs_boot = n_strucs_gen
    else:
        n_strucs_boot = args.n_strucs_boot
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    bd_boot_avg_all = []   # Stores the average bond angle distribution of each bootstrap sample
    eu_dist_boot_all = []   # Stores the euclidean distances of each bootstrap sample
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        # Get bootstrap sample and average
        df_boot = df_gen.iloc[boot_struc_indices]
        bd_boot_avg = df_boot.mean(axis=0).to_numpy()
        bd_boot_avg_all.append(bd_boot_avg)
        # Get Euclidean distance
        eu_dist_boot = euclidean(bd_train_avg, bd_boot_avg)
        eu_dist_boot_all.append(eu_dist_boot)
    bd_boot_avg_all = np.array(bd_boot_avg_all)
    
    # Calculate bootstrap confidence interval
    eu_dist = euclidean(bd_train_avg, bd_gen_avg)
    eu_dist_error_low, eu_dist_error_high = get_conf_interval(eu_dist,  eu_dist_boot_all, args.conf_level)
    bd_gen_CI_low = []   # Lower bound of the gen dataset bond angle distribution confidence interval
    bd_gen_CI_high = []   # Upper bound of the gen dataset bond angle distribution confidence interval
    for i in range(len(bd_boot_avg_all[0])):
        boot_data = bd_boot_avg_all[:, i]
        bd_error_low, bd_error_high = get_conf_interval(bd_gen_avg[i], boot_data, args.conf_level)
        bd_gen_CI_low.append(bd_gen_avg[i] - bd_error_low)
        bd_gen_CI_high.append(bd_gen_avg[i] + bd_error_high)
    
    # Save results
    df_metric = pd.DataFrame.from_dict({'metric': ['Euclidean distance'],
                                        'value': [eu_dist],
                                        'lower error': [eu_dist_error_low],
                                        'upper error': [eu_dist_error_high]
                                        }, 
                                       orient='columns')
    df_metric.to_csv('a_Si_bond_angle_metric.csv', index=False)
    df_bd = pd.DataFrame.from_dict({'Angle': df_train.columns.to_numpy(),
                                    'Train set avg %': bd_train_avg*100,
                                    'Gen set avg %': bd_gen_avg*100,
                                    'Gen set lower bound': np.array(bd_gen_CI_low)*100,
                                    'Gen set upper bound': np.array(bd_gen_CI_high)*100
                                    }, 
                                   orient='columns')
    df_bd.to_csv('avg_bond_angle_distribution.csv', index=False)


if  __name__ == '__main__':
    main()
