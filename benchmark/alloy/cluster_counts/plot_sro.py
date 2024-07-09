import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def transform_df(df, label):
    "Transforms a dataframe for violin plots"
    df_1NN = df.loc[:, df.columns.str.startswith('1NN')]
    df_1NN.columns = df_1NN.columns.str.removeprefix('1NN_')
    df_2NN = df.loc[:, df.columns.str.startswith('2NN')]
    df_2NN.columns = df_2NN.columns.str.removeprefix('2NN_')
    
    df_1NN = df_1NN.melt(var_name='atom pair', value_name='SRO')
    df_1NN['hue'] = label
    df_2NN = df_2NN.melt(var_name='atom pair', value_name='SRO')
    df_2NN['hue'] = label
    
    return df_1NN, df_2NN


parser = argparse.ArgumentParser()
parser.add_argument('--gen_data', type=str, default='cluster_sro.csv', help='path to SRO of generated structures (csv file)')
parser.add_argument('--train_data', type=str, default='/path/to/data/dismai_bench_train_ref_data/alloy_300K_narrow/train_cluster_sro.csv', help='path to SRO of training structures (csv file)')

args = parser.parse_args()

df1 = pd.read_csv(args.train_data, index_col=0)
df2 = pd.read_csv(args.gen_data, index_col=0)

df1_1NN, df1_2NN = transform_df(df1, 'train')
df2_1NN, df2_2NN = transform_df(df2, 'gen')
df_1NN = pd.concat([df1_1NN, df2_1NN])
df_2NN = pd.concat([df1_2NN, df2_2NN])

fig1, ax1 = plt.subplots(1, figsize=(5, 4))
sns.violinplot(ax=ax1, data=df_1NN, y='atom pair', x='SRO', hue='hue', 
               split=True, inner='quart', orient='h',
               linewidth=0.75, gap=0.12, density_norm='count')
ax1.vlines(0, -1, 6, linestyles=':', color='gray', zorder=0)
ax1.set_ylim([5.5, -0.5])
ax1.tick_params(axis='both', labelsize=12)
ax1.set_xlabel('Warren-Cowley SRO parameter', fontsize=12)
ax1.set_ylabel('atom pair', fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels, fontsize=12)
ax1.set_title('1NN', fontsize=13)
fig1.tight_layout()
fig1.savefig('SRO_1NN.png', dpi=200)

fig2, ax2 = plt.subplots(1, figsize=(5, 4))
sns.violinplot(ax=ax2, data=df_2NN, y='atom pair', x='SRO', hue='hue', 
               split=True, inner='quart', orient='h',
               linewidth=0.75, gap=0.12, density_norm='count')
ax2.vlines(0, -1, 6, linestyles=':', color='gray', zorder=0)
ax2.set_ylim([5.5, -0.5])
ax2.tick_params(axis='both', labelsize=12)
ax2.set_xlabel('Warren-Cowley SRO parameter', fontsize=12)
ax2.set_ylabel('atom pair', fontsize=12)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels, fontsize=12)
ax2.set_title('2NN', fontsize=13)
fig2.tight_layout()
fig2.savefig('SRO_2NN.png', dpi=200)

