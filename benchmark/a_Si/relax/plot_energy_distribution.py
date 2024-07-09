import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tck
import argparse
from distutils.util import strtobool

def get_bin_heights(series, bin_edges):
    """
    Gets the bin heights for a distribution

    Parameters
    ----------
    df : pandas.Series to be binned
    bin_edges: numpy.array of the bin edges

    Returns
    -------
    bin_heights: numpy.array of the bin heights in percentage

    """
    bin_heights = pd.cut(series, bin_edges, include_lowest=True)
    bin_heights = bin_heights.value_counts(normalize=True, sort=False).to_numpy()
    
    return bin_heights*100


parser = argparse.ArgumentParser()
parser.add_argument('--gen_data', type=str, default='relax_results.csv', help='path to relaxation results of generated structures (csv file)')
parser.add_argument('--train_data', type=str, default='/path/to/data/dismai_bench_train_ref_data/a_Si/train_energy.csv', help='path to energies of training structures (csv file)')
parser.add_argument('--bin_width', type=float, default=5, help='bin width in meV/atom')
parser.add_argument('--min_bin_edge', type=float, default=-50, help='minimum bin edge in meV/atom')
parser.add_argument('--max_bin_edge', type=float, default=100, help='maximum bin edge in meV/atom')
parser.add_argument('--plot_unrelaxed_energy', default=False, type=lambda x:bool(strtobool(x)), 
                    help='if true, also plots the energy distribution of the unrelaxed structures')

args = parser.parse_args()

bin_edges = np.arange(args.min_bin_edge, args.max_bin_edge + args.bin_width, 
                     args.bin_width)

# Training dataset
df_train = pd.read_csv(args.train_data)
E_train_mean = df_train['energy_per_atom'].mean()
df_train = (df_train['energy_per_atom'] - E_train_mean)*10**3
bin_heights_train = get_bin_heights(df_train, bin_edges)

# Generated dataset
df_gen = pd.read_csv(args.gen_data)
df_gen_initial = (df_gen['E_initial (eV/atom)'] - E_train_mean)*10**3
bin_heights_gen_initial = get_bin_heights(df_gen_initial, bin_edges)
df_gen_final = (df_gen['E_final (eV/atom)'] - E_train_mean)*10**3
bin_heights_gen_final = get_bin_heights(df_gen_final, bin_edges)

# Plot histogram
fig, ax = plt.subplots(1, figsize=(6, 4))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
ax.plot(bin_centers, bin_heights_train, label='train', 
        color='k', linewidth=1.5)
if args.plot_unrelaxed_energy:
    ax.plot(bin_centers, bin_heights_gen_initial, label='gen (unrelaxed)', 
            color='tab:blue', linewidth=1.5)
ax.plot(bin_centers, bin_heights_gen_final, label='gen (relaxed)', 
    color='tab:green', linewidth=1.5)
ax.vlines(0, 0, np.nanmax(np.array([bin_heights_train, bin_heights_gen_initial, bin_heights_gen_final])), 
          linestyles='--', color='k')
ax.grid()
ax.set_xlabel('Normalized energy (meV/atom)', fontsize=14)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_xlim([bin_edges[0], bin_edges[-1]])
ax.tick_params(axis='both', labelsize=13)
ax.xaxis.set_minor_locator(tck.MultipleLocator(args.bin_width))
ax.tick_params('both', length=8, which='major')
ax.tick_params('both', length=4, which='minor')
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig('energy_distribution.png', dpi=200)

