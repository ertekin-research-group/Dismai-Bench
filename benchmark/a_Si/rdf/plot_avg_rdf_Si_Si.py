import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_plot', type=str, default='avg_rdf_Si_Si.csv', help='path to average rdf (csv file)')

args = parser.parse_args()
df = pd.read_csv(args.data_plot)

fig, ax = plt.subplots(1, figsize=(5, 3))
ax.plot(df['r'], df['Train set avg rdf'],
        label='train', color='k', alpha=0.5)
ax.fill_between(df['r'], df['Gen set lower bound'], df['Gen set upper bound'], 
                label='gen', facecolor='tab:blue', edgecolor='tab:blue', alpha=0.8)
ax.set_xlabel('$r$ (Å)', fontsize=14)
ax.set_ylabel('g($r$)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim([-1, 11])
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('avg_rdf_Si_Si.png', dpi=200)


