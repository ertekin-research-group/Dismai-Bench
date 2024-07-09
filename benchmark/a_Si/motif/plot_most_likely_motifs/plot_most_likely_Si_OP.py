import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_plot', type=str, default='most_likely_Si_OP.csv', help='path to most likely order parameter csv file')

args = parser.parse_args()
df = pd.read_csv('most_likely_Si_OP.csv', index_col=0)

fig, ax = plt.subplots(figsize=(6, 4))
df['Train set (%)'].plot.barh(ax=ax, position=0, width=0.4, 
                              color='black', edgecolor='k', linewidth=0.5)
df['Gen set (%)'].plot.barh(ax=ax, position=1, width=0.4, 
                            color='tab:blue', edgecolor='k', linewidth=0.5,
                            xerr=np.stack((df['Gen set lower error (%)'], 
                                           df['Gen set upper error (%)'])), 
                            error_kw=dict(lw=1, capsize=3, capthick=1, ecolor='k'))

ax.set_xlim([0, 100])
ax.set_ylim([-0.65, 5.65])
ax.set_xlabel('Percentage (%)', fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(['train', 'gen'], fontsize=12, loc='lower right')
fig.tight_layout()
fig.savefig('Si_coord_env.png', dpi=200)



