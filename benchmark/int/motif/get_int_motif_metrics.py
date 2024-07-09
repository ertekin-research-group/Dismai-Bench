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

def run_bootstrap(args, atom_type):
    """
    Calculates the bootstrap confidence intervals of the metrics

    Parameters
    ----------
    args: argparse.Namespace object containing parameters
    atom_type : string literal of atom type ('Li', 'Co' or 'Sc')

    Returns
    -------
    fingerprint_avg_dict: dictionary of average fingerprints for 
        train set, gen set, and all bootstrap samples
    metric_dict: dictionary of all metrics with their respective 
        lower and upper errors

    """
    print('Running bootstrapping for ' + atom_type)
    
    if atom_type == 'Li':
        n_atoms = args.n_atoms_Li
    if atom_type == 'Co':
        n_atoms = args.n_atoms_Co
    if atom_type == 'Sc':
        n_atoms = args.n_atoms_Sc
    
    df_train = pd.read_csv(args.data_train_dir + '/cnn_stats_' + atom_type + '.csv')
    df_gen = pd.read_csv(args.data_gen_dir + '/cnn_stats_' + atom_type + '.csv')
    
    # Get average fingerprint of train dataset
    fingerprint_avg_train = df_train.iloc[:, 8:117].mean().to_numpy()
    
    # Get fingerprint of each structure in gen dataset (used for bootstrapping)
    fingerprint_gen_all = []
    n_strucs_gen = df_gen['structure_number'].iloc[-1]   # Number of structures in gen dataset
    for i in range(n_strucs_gen):
        atom_indices = list(range(i*n_atoms, (i+1)*n_atoms))
        df_gen_struc = df_gen.iloc[atom_indices]
        # Average fingerprints of all atoms in the structure
        fingerprint = df_gen_struc.iloc[:, 8:117].mean().to_numpy()
        fingerprint_gen_all.append(fingerprint)
    fingerprint_gen_all = np.array(fingerprint_gen_all)
    # Get average fingerprint of gen dataset
    fingerprint_avg_gen = np.mean(fingerprint_gen_all, axis=0)
    
    # Perform bootstrapping
    rng = np.random.default_rng(args.seed)
    gen_struc_indices = np.arange(n_strucs_gen)   # Indices of structures in the gen dataset
    fingerprint_avg_boot_all = []   # Stores the average fingerprint of each bootstrap sample
    eu_dist_boot_all = []   # Stores the euclidean distances of each bootstrap sample 
    cos_dist_boot_all = []   # Stores the cosine distance of each bootstrap sample
    if args.n_strucs_boot < 0:
        n_strucs_boot = df_gen['structure_number'].iloc[-1]
    else:
        n_strucs_boot = args.n_strucs_boot
    for num in tqdm(range(args.n_reps)):
        # Obtain sampling indices for bootstrap sample
        boot_struc_indices = rng.choice(gen_struc_indices, size=n_strucs_boot, replace=True)
        # Get fingerprints of all structures in bootstrap sample
        fingerprint_boot_all = fingerprint_gen_all[boot_struc_indices]
        # Average the fingerprints in the bootstrap sample
        fingerprint_avg_boot = np.mean(fingerprint_boot_all, axis=0)
        fingerprint_avg_boot_all.append(fingerprint_avg_boot)
        eu_dist_boot = euclidean(fingerprint_avg_train, fingerprint_avg_boot)
        cos_dist_boot = cosine(fingerprint_avg_train, fingerprint_avg_boot)
        eu_dist_boot_all.append(eu_dist_boot)
        cos_dist_boot_all.append(cos_dist_boot)

    # Calculate bootstrap confidence interval
    eu_dist = euclidean(fingerprint_avg_train, fingerprint_avg_gen)
    cos_dist = cosine(fingerprint_avg_train, fingerprint_avg_gen)
    eu_dist_error_low, eu_dist_error_high = get_conf_interval(eu_dist, eu_dist_boot_all, args.conf_level)
    cos_dist_error_low, cos_dist_error_high = get_conf_interval(cos_dist, cos_dist_boot_all, args.conf_level)
    
    # Create dictionaries of results
    fingerprint_avg_dict = {'train': fingerprint_avg_train,
                            'gen': fingerprint_avg_gen,
                            'boot_all': np.array(fingerprint_avg_boot_all)
                            }
    metric_dict = {'eu_dist': (eu_dist, eu_dist_error_low, eu_dist_error_high),
                   'cos_dist': (cos_dist, cos_dist_error_low, cos_dist_error_high)
                   }
    
    return fingerprint_avg_dict, metric_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_gen_dir', type=str, default='.', help='path to directory containing coordination motif csv files of generated structures')
    parser.add_argument('--data_train_dir', type=str, default='/path/to/data/dismai_bench_train_ref_data/int/train_motif', help='path to directory containing coordination motif csv files of training structures')
    parser.add_argument('--n_strucs_boot', type=int, default=-1, help='number of structures in each bootstrap sample (if < 0, set to be equal to number of structures in gen dataset')
    parser.add_argument('--n_reps', type=int, default=1000, help='number of bootstrap repetitions')
    parser.add_argument('--conf_level', type=float, default=0.95, help='confidence level for confidence interval')
    parser.add_argument('--n_atoms_Li', type=int, default=72, help='number of Li atoms in each structure')
    parser.add_argument('--n_atoms_Co', type=int, default=36, help='number of Co atoms in each structure')
    parser.add_argument('--n_atoms_Sc', type=int, default=12, help='number of Sc atoms in each structure')
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed for bootstrapping')
    parser.add_argument('--n_strucs_ori', type=int, default=1000, help='number of structures originally generated using the generative model (to calculate the percentage of failed structures)')
    
    args = parser.parse_args()
    
    # Get metrics for each atom type
    fingerprint_avg_Li, metric_Li = run_bootstrap(args, 'Li')
    fingerprint_avg_Co, metric_Co = run_bootstrap(args, 'Co')
    fingerprint_avg_Sc, metric_Sc = run_bootstrap(args, 'Sc')
    
    # Get metrics for all atom types combined
    n_atoms_total = args.n_atoms_Li + args.n_atoms_Co + args.n_atoms_Sc   # Total number of atoms with calculated fingerprints in each structure
    fingerprint_avg_train = (fingerprint_avg_Li['train']*args.n_atoms_Li + 
                             fingerprint_avg_Co['train']*args.n_atoms_Co + 
                             fingerprint_avg_Sc['train']*args.n_atoms_Sc) / n_atoms_total
    fingerprint_avg_gen = (fingerprint_avg_Li['gen']*args.n_atoms_Li + 
                           fingerprint_avg_Co['gen']*args.n_atoms_Co + 
                           fingerprint_avg_Sc['gen']*args.n_atoms_Sc) / n_atoms_total
    fingerprint_avg_boot_all = (fingerprint_avg_Li['boot_all']*args.n_atoms_Li + 
                                fingerprint_avg_Co['boot_all']*args.n_atoms_Co + 
                                fingerprint_avg_Sc['boot_all']*args.n_atoms_Sc) / n_atoms_total
    eu_dist_boot_all = []   # Stores the euclidean distances of each bootstrap sample 
    cos_dist_boot_all = []   # Stores the cosine distance of each bootstrap sample
    for fingerprint_avg_boot in fingerprint_avg_boot_all:
        eu_dist_boot = euclidean(fingerprint_avg_train, fingerprint_avg_boot)
        cos_dist_boot = cosine(fingerprint_avg_train, fingerprint_avg_boot)
        eu_dist_boot_all.append(eu_dist_boot)
        cos_dist_boot_all.append(cos_dist_boot)
    eu_dist = euclidean(fingerprint_avg_train, fingerprint_avg_gen)
    cos_dist = cosine(fingerprint_avg_train, fingerprint_avg_gen)
    eu_dist_error_low, eu_dist_error_high = get_conf_interval(eu_dist, eu_dist_boot_all, args.conf_level)
    cos_dist_error_low, cos_dist_error_high = get_conf_interval(cos_dist, cos_dist_boot_all, args.conf_level) 
    
    # Add metric results for each atom type
    metric_names = []
    metric_values = []
    metric_lower_errors = []
    metric_upper_errors = []
    for item in [(metric_Li, 'Li'), (metric_Co, 'Co'), (metric_Sc, 'Sc')]:
        metric_dict = item[0]
        atom_type = item[1]
        metric_names += ['Euclidean distance '+atom_type, 'Cosine distance '+atom_type]
        metric_values += [metric_dict['eu_dist'][0], metric_dict['cos_dist'][0]]
        metric_lower_errors += [metric_dict['eu_dist'][1], metric_dict['cos_dist'][1]]
        metric_upper_errors += [metric_dict['eu_dist'][2], metric_dict['cos_dist'][2]]
    # Add metric results for all atom types combined
    metric_names += ['Euclidean distance all', 'Cosine distance all']
    metric_values += [eu_dist, cos_dist]
    metric_lower_errors += [eu_dist_error_low, cos_dist_error_low]
    metric_upper_errors += [eu_dist_error_high, cos_dist_error_high]
    # Add metric results for percentage of failed structures
    df_gen = pd.read_csv(args.data_gen_dir + '/cnn_stats_Sc.csv')
    percent_failed_strucs = 100.0 - (df_gen['structure_number'].iloc[-1] / args.n_strucs_ori * 100)
    assert percent_failed_strucs <= 100.0, 'Percentage of failed structures calculated to be > 100 %, please check that n_strucs_ori is set correctly'
    metric_names += ['Failed structures (%)']
    metric_values += [percent_failed_strucs]
    metric_lower_errors += [None]
    metric_upper_errors += [None]
    # Save results
    df_results = pd.DataFrame.from_dict({'metric': metric_names,
                                         'value': metric_values,
                                         'lower error': metric_lower_errors,
                                         'upper error': metric_upper_errors}, 
                                        orient='columns')
    df_results.to_csv('int_motif_metrics.csv', index=False)


if  __name__ == '__main__':
    main()
