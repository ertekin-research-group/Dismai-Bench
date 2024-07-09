import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_plot', type=str, default='avg_bond_angle_distribution.csv', help='path to average bond angle distribution (csv file)')

args = parser.parse_args()
df = pd.read_csv(args.data_plot)

fig, ax = plt.subplots(1, figsize=(5, 3))
ax.plot(df['Angle'], df['Train set avg %'],
        label='train', color='k', alpha=0.5)
ax.fill_between(df['Angle'], df['Gen set lower bound'], df['Gen set upper bound'], 
                label='gen', facecolor='tab:blue', edgecolor='tab:blue', alpha=0.8)
ax.set_xlabel('Angle (°)', fontsize=14)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim([0, 180])
ax.set_ylim([-0.1, 1.7])
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('avg_bond_angle_distribution.png', dpi=200)


