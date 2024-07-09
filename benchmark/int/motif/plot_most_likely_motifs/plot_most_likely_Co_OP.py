import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import matplotlib.patches as patches


parser = argparse.ArgumentParser()
parser.add_argument('--data_plot', type=str, default='most_likely_Co_OP.csv', help='path to most likely order parameter csv file')

args = parser.parse_args()
df = pd.read_csv(args.data_plot, index_col=0)

# Plot most likely Sc order parameters
fig1, ax1 = plt.subplots(figsize=(6, 3.5))
df['Train set (%)'].plot.barh(ax=ax1, position=0, width=0.4, 
                              color='black', edgecolor='k', linewidth=0.5)
df['Gen set (%)'].plot.barh(ax=ax1, position=1, width=0.4, 
                            color='tab:blue', edgecolor='k', linewidth=0.5,
                            xerr=np.stack((df['Gen set lower error (%)'], 
                                           df['Gen set upper error (%)'])), 
                            error_kw=dict(lw=1, capsize=3, capthick=1, ecolor='k'))
ax1.set_xlim([0, 100])
ax1.set_ylim([-0.65, 4.65])
ax1.set_xlabel('Percentage (%)', fontsize=13)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(['train', 'gen'], fontsize=11)
ax1.set_title('Co motifs', fontsize=12)
fig1.tight_layout()
fig1.savefig('Co_coord_env.png', dpi=200)

# Plot most likely Sc order parameters subdivided based on number of Cl neighbors
max_cn = int(df.index[0].split('_')[-1])   # Maximum coordination number in filtered order parameters
my_cmap = ['#E03524','#F07C12','#FFC200','#90BC1A','#0095AC','#5050B9','#903498','grey']
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
df.iloc[:, 4:4+max_cn+1].plot.barh(ax=ax2, stacked=True, position=0, width=0.4, 
                                   edgecolor='k', linewidth=0.5, color=my_cmap)
df['Train set (%)'].plot.barh(ax=ax2, color='none', position=0, width=0.4)
df.iloc[:, 4+max_cn+1:4+2*max_cn+2].plot.barh(ax=ax2, stacked=True, position=1, width=0.4, 
                                              edgecolor='k', linewidth=0.5, color=my_cmap, 
                                              hatch='///', legend=False)
df['Gen set (%)'].plot.barh(ax=ax2, color='none', position=1, width=0.4,
                            xerr=np.stack((df['Gen set lower error (%)'], 
                                           df['Gen set upper error (%)'])), 
                            error_kw=dict(lw=1, capsize=3, capthick=1, ecolor='k'))
ax2.set_xlim([0, 100])
ax2.set_ylim([-0.65, 4.65])
ax2.set_xlabel('Percentage (%)', fontsize=13)
ax2.tick_params(axis='both', which='major', labelsize=12)
handles = []
for n_Cl in range(max_cn+1):
    handles.append(str(n_Cl) + ' Cl')
legend_main = ax2.legend(handles, fontsize=11, loc='upper right')
ax2.add_artist(legend_main)
box_train = patches.Patch(label='train', facecolor='w', edgecolor='k')
box_gen = patches.Patch(label='gen', facecolor='w', edgecolor='k', hatch='///')
legend_supp = ax2.legend(handles=[box_train, box_gen], fontsize=11, loc='lower right')
ax2.add_artist(legend_supp)
ax2.set_title('Co motifs', fontsize=12)
fig2.tight_layout()
fig2.savefig('Co_coord_env_with_n_Cl.png', dpi=200)


