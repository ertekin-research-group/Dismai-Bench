from glob import glob
import argparse
import shutil
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    
    args = parser.parse_args()
    
    # Get the number of batches (note: batch number starts from 1)
    batch_dir_names = glob('batch_*')
    batch_numbers = [dir_name.split('_')[-1] for dir_name in batch_dir_names]
    batch_numbers.sort()
    last_batch_number = int(batch_numbers[-1])
    assert last_batch_number > 1, 'Only 1 batch found, this script only works if there are more than 1 batches'
    
    # Get the total number of structures that should have been relaxed
    data_fname = sorted(glob('*.extxyz'))[0]
    n_strucs = 0
    with open(data_fname, 'r') as f:
        contents = f.readlines()
        for line in contents:
            n_strucs += len(re.findall('Lattice', string=line))
    
    # Check that relaxations of all structures have completed
    batch_dnf = []   # Stores the batch numbers of batches that did not finish relaxations
    for i in range(1, last_batch_number+1):
        # Determine the correct last structure number if all relaxations have completed
        if i == last_batch_number:
            correct_last_struc_num = n_strucs
        else:
            correct_last_struc_num = i * args.n_strucs_per_batch
        # Check the structure number of the last successful structure
        with open('batch_'+str(i)+'/relax_results.csv', 'r') as f:
            last_line = f.readlines()[-1]
            last_struc_num = int(last_line.split(',')[1])
            if last_struc_num != correct_last_struc_num:
                # Check the structure number of the last failed structure
                with open('batch_'+str(i)+'/discarded.csv', 'r') as f:
                    last_line = f.readlines()[-1]
                    last_struc_num = int(last_line.split(',')[1])
                    if last_struc_num != correct_last_struc_num:
                        batch_dnf.append(i)
    if len(batch_dnf) > 0:
        raise Exception(f'Batch {batch_dnf} did not finish relaxations, please restart the relaxations.')
    
    # Collect relaxation results (.csv)
    shutil.copyfile('batch_1/relax_results.csv', 'relax_results.csv')
    with open('relax_results.csv', 'a') as file1:
        for i in range(2, last_batch_number+1):
            with open('batch_'+str(i)+'/relax_results.csv', 'r') as file2:
                next(file2)   # skip header
                shutil.copyfileobj(file2, file1)
    
    # Collect relaxed structures (.extxyz)
    shutil.copyfile('batch_1/gen_relaxed.extxyz', 'gen_relaxed.extxyz')
    with open('gen_relaxed.extxyz', 'a') as file1:
        for i in range(2, last_batch_number+1):
            with open('batch_'+str(i)+'/gen_relaxed.extxyz', 'r') as file2:
                shutil.copyfileobj(file2, file1)
    

if  __name__ == '__main__':
    main()