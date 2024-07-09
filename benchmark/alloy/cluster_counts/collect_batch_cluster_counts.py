import json
import pandas as pd
import numpy as np
from glob import glob
from ase.io import read


def main():
    # Get the number of batches (note: batch number starts from 1)
    batch_dir_names = glob("batch_*")
    batch_numbers = [dir_name.split('_')[-1] for dir_name in batch_dir_names]
    batch_numbers.sort()
    last_batch_number = int(batch_numbers[-1])
    
    # Collect counts from all batches
    cluster_count_all = []
    vac_count_all = []
    for i in range(1, last_batch_number+1):
        with open('batch_'+str(i)+'/cluster_count.json') as filehandle:
            cluster_count = json.load(filehandle)
            cluster_count_all += cluster_count
        df = pd.read_csv('batch_'+str(i)+'/vac_count.csv', index_col=0)
        vac_count_all += df['vac_count'].to_list()
    
    # Check number of structures collected
    data_fname = glob('*.extxyz')[0]
    data = read(data_fname, index=':', format='extxyz')
    print('{}/{} structures collected'.format(len(cluster_count_all), len(data)))
    if len(cluster_count_all) != len(data):
        raise Exception('There are missing structures in the counts!')
    
    # Save counts
    with open('cluster_count.json', 'w') as filehandle:
        json.dump(cluster_count_all, filehandle)
    df = pd.DataFrame.from_dict({'structure_index': np.arange(0, len(cluster_count_all)),
                                 'vac_count': vac_count_all})
    df.to_csv('vac_count.csv', index=False)


if  __name__ == '__main__':
    main()