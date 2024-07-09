import json
import pandas as pd
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cluster_count.json', help='path to cluster counts (json file)')
    
    args = parser.parse_args()
    with open(args.data, 'r') as filehandle:
        cluster_count_all = json.load(filehandle)
    
    # Create a dictionary for each cluster (0NN, 1NN, 2NN, etc), and put them in a list
    count_dict_all = [{'Cr':[], 'Fe':[], 'Ni':[]}]   # 0NN
    for _ in range(7):   # 1NN to 7NN
        count_dict_all.append({'Cr, Cr':[], 
                               'Cr, Fe':[], 
                               'Cr, Ni':[], 
                               'Fe, Fe':[], 
                               'Fe, Ni':[], 
                               'Ni, Ni':[]})
    # Fill dictionaries with cluster counts of each structure
    for i in range(len(cluster_count_all)):
        cluster_count = cluster_count_all[i]
        for j in range(8):   # 8 clusters (0NN to 7 NN)
            cluster_dict = cluster_count[j][-1]
            n_entries = 0
            for key in count_dict_all[j].keys():
                if key in cluster_dict:
                    count_dict_all[j][key].append(cluster_dict[key])
                    n_entries += 1
                else:
                    count_dict_all[j][key].append(0.)
            assert n_entries == len(cluster_dict), 'Wrong key found in structure index {} cluster index {}'.format(i, j)
            
    # Create DataFrame of cluster counts
    df = pd.DataFrame()
    for j in range(8):   # 8 clusters (0NN to 7 NN)
        for key in count_dict_all[j].keys():
            col_name = str(j) + 'NN_' + key
            if ', ' in col_name:
                col_name = col_name.replace(', ', '-')
            df[col_name] = count_dict_all[j][key]
    
    # Convert counts to probability
    df_prob = df.copy()
    multiplicity = [1, 6, 3, 12, 6, 12, 4, 24]   # multiplicity of each cluster
    col_idx = 0   # Initialize column index for indexing columns
    for j in range(8):   # 8 clusters (0NN to 7 NN)
        if j == 0:   
            for _ in range(3):   # multiplicity of 1, just move to next column
                col_idx += 1
        else:
            for _ in range(6):
                df_prob.iloc[:, col_idx] /= multiplicity[j]
                col_idx += 1
    df_prob.to_csv('cluster_prob.csv', index_label='structure index')
    
    # Calculate formation energy using effective cluster interaction (ECI) of cluster expansion
    eci = np.array([-0.00932958469464057, 
                    -0.0, -0.0, -0.0, 
                    0.0003376140261270978, -0.006093130106866821, -0.017503750379865866, 0.0, -0.0, 0.0022227714241077777,
                    -0.03477817417403486, 0.0, 0.005939741713420254, -0.0006455514901675906, 0.016358092689283846, -0.0,
                    0.0015984460745564193, -0.005292076441853677, -0.003021082122451338, 0.0, -0.0, 0.0,
                    -0.0, 0.0018668024945124595, -0.0038807821841662124, -0.0, -0.0036085744161752594, 0.0012250045279512465,
                    -0.0007252972937068049, 0.0005925701791865664, 0.0028510208730949406, -0.0005707393430416303, 0.0018947957869953978, -0.001669283899958865,
                    0.0, -0.005562649078917254, 0.00172340842194012, 0.0, 0.00015073769660058886, -0.003134770684447135,
                    0.001983244154690138, -0.0018490697130025958, -0.002680779582402635, 0.0003351340012252481, -0.0021377444095509075, 0.0010601861757962314,
                    ])
    df_energy = df.iloc[:, np.arange(len(df.columns))] * eci[1:]
    df_energy = df_energy.sum(axis=1) + eci[0]
    df_energy.to_csv('formation_energy.csv', index_label='structure index', 
                     header=['formation energy (eV/atom)'])
    
    # Calculate Warren-Cowley short-range order parameter for 1NN and 2NN
    df_sro = pd.DataFrame()
    for j in range(1, 2+1):   # 1NN and 2NN
        pf = str(j) + 'NN_'   # column name prefix
        df_sro[pf+'Fe-Ni'] = 1 - 0.5*(df[pf+'Fe-Ni'] / (2*df[pf+'Fe-Fe'] + df[pf+'Cr-Fe'] + df[pf+'Fe-Ni']) / df['0NN_Ni'] + 
                                      df[pf+'Fe-Ni'] / (2*df[pf+'Ni-Ni'] + df[pf+'Cr-Ni'] + df[pf+'Fe-Ni']) / df['0NN_Fe']
                                      )
        df_sro[pf+'Cr-Fe'] = 1 - 0.5*(df[pf+'Cr-Fe'] / (2*df[pf+'Cr-Cr'] + df[pf+'Cr-Ni'] + df[pf+'Cr-Fe']) / df['0NN_Fe'] + 
                                      df[pf+'Cr-Fe'] / (2*df[pf+'Fe-Fe'] + df[pf+'Fe-Ni'] + df[pf+'Cr-Fe']) / df['0NN_Cr']
                                      )
        df_sro[pf+'Cr-Ni'] = 1 - 0.5*(df[pf+'Cr-Ni'] / (2*df[pf+'Cr-Cr'] + df[pf+'Cr-Fe'] + df[pf+'Cr-Ni']) / df['0NN_Ni'] + 
                                      df[pf+'Cr-Ni'] / (2*df[pf+'Ni-Ni'] + df[pf+'Fe-Ni'] + df[pf+'Cr-Ni']) / df['0NN_Cr']
                                      )
        df_sro[pf+'Fe-Fe'] = 1 - 2*df[pf+'Fe-Fe'] / (2*df[pf+'Fe-Fe'] + df[pf+'Fe-Ni'] + df[pf+'Cr-Fe']) / df['0NN_Fe']
        df_sro[pf+'Cr-Cr'] = 1 - 2*df[pf+'Cr-Cr'] / (2*df[pf+'Cr-Cr'] + df[pf+'Cr-Ni'] + df[pf+'Cr-Fe']) / df['0NN_Cr']
        df_sro[pf+'Ni-Ni'] = 1 - 2*df[pf+'Ni-Ni'] / (2*df[pf+'Ni-Ni'] + df[pf+'Fe-Ni'] + df[pf+'Cr-Ni']) / df['0NN_Ni']
    df_sro.to_csv('cluster_sro.csv', index_label='structure index')


if  __name__ == '__main__':
    main()
