import argparse
import torch
import json
from tqdm import tqdm
from ase import Atoms
from ase.io import write
import csv


def unnormalize_tensor(tensor, min_val, max_val):
    """
    Performs min-max unnormalization on torch.Tensor in-place
    
    Parameters
    ----------
    tensor : torch.Tensor
    min_val : int or float
    max_val : int or float
    
    Returns
    -------
    None (in-place operation)
    
    """
    tensor *= (max_val - min_val)
    tensor += min_val
    
    return

class AverageMeter(object):
    "Computes and stores the average and current value"

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def convert_extxyz(unimat_all, param, el_index_gt, write_fname, scaler=None):
    """
    Converts a stacked UniMat to extxyz, and calculates the element species accuracy
    
    Parameters
    ----------
    unimat_all : torch.Tensor
        Stacked UniMat tensor of shape (n_strucs, 3, n_atoms, UniMat_length, UniMat_length)
    param : dict
        Dictionary of parameters regarding the dataset
    el_index_gt : list
        Ground truth element index of shape (n_atoms, 2) 
    write_fname : str
        Filename to write extxyz file
    scaler : dict
        (optional) Dictionary of min and max values used for normalization
    
    Returns
    -------
    None. 
    An extxyz file of the structures and csv file of the element species accuracy
    will be written.
    
    """
    # Intialize structure parameters for ase.Atoms object
    struc_all = []   # For storing all structures
    n_atoms = param['n_atoms']   # Number of atoms (of 1 structure)
    cell = [param['a'], param['b'], param['c'], 
            param['alpha'], param['beta'], param['gamma']]   # Lattice parameters
    el_symbol_map = dict((tuple(value), key) for key, value in param['el_index_map'].items())
    el_symbol_gt = [el_symbol_map[tuple(el_index)] for el_index in el_index_gt]   # Ground truth element symbols
    
    # Count index where each element starts in el_index_gt
    el_start_atom_indexes = []   # Stores the atom index where each element starts
    el_start_el_indexes = []   # Stores the UniMat element index correponding to the element starting
    prev_el = 12345
    for i in range(len(el_index_gt)):
        curr_el = el_index_gt[i]
        if curr_el != prev_el:
            el_start_atom_indexes.append(i)
            el_start_el_indexes.append(curr_el)
        prev_el = curr_el
    n_el = len(param['el_index_map'])   # Number of elements
    assert len(el_start_atom_indexes) == n_el, 'Element species of reference UniMat are not sorted'
    
    # Initialize meter for element species accuracy
    meter_el = AverageMeter()
    
    for i in tqdm(range(unimat_all.shape[0])):
        unimat = unimat_all[i]
        
        # Get element species accuracy
        el_acc = []
        for atom_idx in range(unimat.shape[1]):
            # Get UniMat slice of the current atom, and round to nearest integer
            unimat_atom_slice = unimat[:, atom_idx].round()
            # Patch the correct element index with -1.
            unimat_atom_slice[:, 
                              el_index_gt[atom_idx][0], 
                              el_index_gt[atom_idx][1]] = -1.
            # Check if element species is accurate (element is considered absent if either x, y, or z is -1.; here all should be absent)
            if torch.all(unimat_atom_slice.min(dim=0)[0] == -1.):
                el_acc.append(1)
            else:
                el_acc.append(0)
        el_acc = sum(el_acc) / len(el_acc)
        meter_el.update(el_acc)
        
        # Construct structure if element species of all atoms are accurate
        if el_acc == 1.:
            # Get coordinates
            coords = []
            for el_num in range(n_el-1):
                coords_el = unimat[:,
                                   el_start_atom_indexes[el_num]:el_start_atom_indexes[el_num+1],
                                   el_start_el_indexes[el_num][0],
                                   el_start_el_indexes[el_num][1]]
                coords_el = torch.movedim(coords_el, 0, 1)
                coords.append(coords_el)
            coords_el = unimat[:,
                               el_start_atom_indexes[-1]:n_atoms,
                               el_start_el_indexes[-1][0],
                               el_start_el_indexes[-1][1]]
            coords_el = torch.movedim(coords_el, 0, 1)
            coords.append(coords_el)
            coords = torch.cat(coords, dim=0)
            
            # Unnormalize coordinates (if applicable)
            if scaler is not None:
                unnormalize_tensor(coords[:, 0], scaler['coord_x']['min'], scaler['coord_x']['max'])
                unnormalize_tensor(coords[:, 1], scaler['coord_y']['min'], scaler['coord_y']['max'])
                unnormalize_tensor(coords[:, 2], scaler['coord_z']['min'], scaler['coord_z']['max']) 
            
            # Construct structure
            struc = Atoms(symbols=el_symbol_gt, 
                          scaled_positions=coords.numpy(),
                          cell=cell,
                          pbc=True)
            struc_all.append(struc)
        else:
            print(f'Ignoring structure index {i} due to incorrect element species')
    
    # Save structures and statistics
    write(write_fname, struc_all, format='extxyz')
    with open('element_accuracy.csv', "w", newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['element_accuracy', 'fraction_structures_correct'])
        csvwriter.writerow([meter_el.avg, len(struc_all)/unimat_all.shape[0]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unimat_data', type=str, default='unimat_gen.pt', help='path to UniMat (pt file)')
    parser.add_argument('--train_data_dir', type=str, default='/path/to/data/train_val_data/unimat/int', help='path to directory containing UniMat train/val datasets and json files')
    parser.add_argument('--write_fname', type=str, default='gen.extxyz', help='filename to write extxyz file')
    
    args = parser.parse_args()
    
    with open(args.train_data_dir + '/param.json') as f:
        param = json.load(f)
    with open(args.train_data_dir + '/scaler.json') as f:
        scaler = json.load(f)
    # Get ground truth element indexes in UniMat
    unimat_ref = torch.load(args.train_data_dir + '/unimat_val.pt')
    el_index_gt = torch.empty((param['n_atoms'], 2), dtype=torch.int)
    el_index_gt[:, 0] = (torch.argmax(unimat_ref[0, 0, :], dim=1)).max(dim=1)[0]
    el_index_gt[:, 1] = (torch.argmax(unimat_ref[0, 0, :], dim=2)).max(dim=1)[0]
    el_index_gt = el_index_gt.tolist()
    
    print('Loading UniMat...')
    unimat_all = torch.load(args.unimat_data)
    print('Converting to extxyz...')
    convert_extxyz(unimat_all, param, el_index_gt, args.write_fname, scaler)


if __name__ == '__main__':
    main()



