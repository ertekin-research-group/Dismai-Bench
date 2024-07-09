import argparse
import torch
import json
from tqdm import tqdm
from ase import Atoms
from ase.io import write
import csv


def unnormalize_array(array, min_val, max_val):
    """
    Performs min-max unnormalization on an array
    
    Parameters
    ----------
    array : np.array
    min_val : int or float
    max_val : int or float
    
    Returns
    -------
    array : np.array
    
    """
    array *= (max_val - min_val)
    array += min_val
    
    return array

def unnormalize_stacked_crys_tens(stacked_crys_tens, param, scaler):
    """
    Performs min-max unnormalization on a stacked CrysTens array
    
    Parameters
    ----------
    stacked_crys_tens : np.array
        Stacked CrysTens array of shape (n_strucs, 4, CrysTens_length, CrysTens_length)
    param : dict
        Dictionary of parameters regarding the dataset
    scaler: dict
        Dictionary of min and max values used for normalization    
    
    Returns
    -------
    stacked_crys_tens: np.array
        Unnormalized stacked CrysTens array
    
    """
    if not param['fixed_param']:
        raise Exception('Error: This code does not support datasets with varying composition \
                        and/or number of atoms')
    
    skip = 11 + param['padding']   # Number of lines to skip

    stacked_crys_tens[:, :, 0, skip:] = unnormalize_array(stacked_crys_tens[:, :, 0, skip:],
                                                          scaler['atom_num']['min'],
                                                          scaler['atom_num']['max'])
    stacked_crys_tens[:, :, skip:, 0] = unnormalize_array(stacked_crys_tens[:, :, skip:, 0],
                                                          scaler['atom_num']['min'],
                                                          scaler['atom_num']['max'])
    
    stacked_crys_tens[:, :, 1, skip:] = unnormalize_array(stacked_crys_tens[:, :, 1, skip:],
                                                          scaler['coord_x']['min'],
                                                          scaler['coord_x']['max'])
    stacked_crys_tens[:, :, skip:, 1] = unnormalize_array(stacked_crys_tens[:, :, skip:, 1],
                                                          scaler['coord_x']['min'],
                                                          scaler['coord_x']['max'])
    
    stacked_crys_tens[:, :, 2, skip:] = unnormalize_array(stacked_crys_tens[:, :, 2, skip:],
                                                          scaler['coord_y']['min'],
                                                          scaler['coord_y']['max'])
    stacked_crys_tens[:, :, skip:, 2] = unnormalize_array(stacked_crys_tens[:, :, skip:, 2],
                                                          scaler['coord_y']['min'],
                                                          scaler['coord_y']['max'])
    
    stacked_crys_tens[:, :, 3, skip:] = unnormalize_array(stacked_crys_tens[:, :, 3, skip:],
                                                          scaler['coord_z']['min'],
                                                          scaler['coord_z']['max'])
    stacked_crys_tens[:, :, skip:, 3] = unnormalize_array(stacked_crys_tens[:, :, skip:, 3],
                                                          scaler['coord_z']['min'],
                                                          scaler['coord_z']['max'])
    
    stacked_crys_tens[:, 0, skip:, skip:] = unnormalize_array(stacked_crys_tens[:, 0, skip:, skip:],
                                                              scaler['dist_norm']['min'],
                                                              scaler['dist_norm']['max'])
    
    stacked_crys_tens[:, 1, skip:, skip:] = unnormalize_array(stacked_crys_tens[:, 1, skip:, skip:],
                                                              scaler['dist_dx']['min'],
                                                              scaler['dist_dx']['max'])
    
    stacked_crys_tens[:, 2, skip:, skip:] = unnormalize_array(stacked_crys_tens[:, 2, skip:, skip:],
                                                              scaler['dist_dy']['min'],
                                                              scaler['dist_dy']['max'])
    
    stacked_crys_tens[:, 3, skip:, skip:] = unnormalize_array(stacked_crys_tens[:, 3, skip:, skip:],
                                                              scaler['dist_dz']['min'],
                                                              scaler['dist_dz']['max'])
    
    return stacked_crys_tens

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

def convert_extxyz(stacked_crys_tens, param, atom_num_gt, write_fname):
    """
    Converts a stacked CrysTens array to extxyz, and calculates the statistics of errors
    
    Parameters
    ----------
    stacked_crys_tens : torch.Tensor
        Stacked CrysTens tensor of shape (n_strucs, 4, CrysTens_length, CrysTens_length)
    param : dict
        Dictionary of parameters regarding the dataset
    atom_num_gt : torch.Tensor
        Ground truth atomic number of shape (n_atoms) 
    write_fname : str
        Filename to write extxyz file
    
    Returns
    -------
    None. 
    An extxyz file of the structures and csv file of the statistics will be written.
    
    """
    skip = 11 + param['padding']   # Number of lines to skip
    stacked_crys_tens = torch.movedim(stacked_crys_tens, 1, -1)
    
    # Intialize structure parameters for ase.Atoms object
    cell = [param['a'], param['b'], param['c'], 
            param['alpha'], param['beta'], param['gamma']]   # Lattice parameters
    atom_el_map = dict((value, key) for key, value in param['atom_num_map'].items())
    atom_el_gt = [atom_el_map[atom_num.item()] for atom_num in atom_num_gt]   # Ground truth element symbols
    struc_all = []   # For storing all structures
    
    # Initialize meters for saving statistics
    meter_atom_num = AverageMeter()
    meter_lat_len = AverageMeter()
    meter_lat_angle = AverageMeter()
    meter_spacegroup = AverageMeter()
    meter_coord = AverageMeter()
    meter_dist = AverageMeter()
    l1loss = torch.nn.L1Loss()
    
    for i in tqdm(range(stacked_crys_tens.shape[0])):
        crys_tens = stacked_crys_tens[i]
        
        # Atomic numbers
        atom_num = torch.cat((crys_tens[0, skip:, :], crys_tens[skip:, 0, :]),
                             dim=1)
        atom_num_avg = atom_num.mean(dim=1)
        atom_num_avg = atom_num_avg.round().int()
        atom_num_acc = (atom_num_avg == atom_num_gt).float().mean()   # Accuracy of atomic number
        meter_atom_num.update(atom_num_acc.item())
        
        # Lattice lengths
        lat_len = torch.cat((crys_tens[4:6+1, skip:, :], 
                             torch.movedim(crys_tens[skip:, 4:6+1, :], 0, 1)),
                            dim=2)
        lat_len_mae = lat_len.abs().mean()   # MAE of lattice lengths
        meter_lat_len.update(lat_len_mae.item())
        
        # Lattice angles
        lat_angle = torch.cat((crys_tens[7:9+1, skip:, :], 
                               torch.movedim(crys_tens[skip:, 7:9+1, :], 0, 1)),
                              dim=2)
        lat_angle_mae = lat_angle.abs().mean()   # MAE of lattice angles
        meter_lat_angle.update(lat_angle_mae.item())
        
        # Space group
        spacegroup = torch.cat((crys_tens[10, skip:, :], crys_tens[skip:, 10, :]),
                             dim=1)
        spacegroup_mae = spacegroup.abs().mean()   # MAE of spacegroup
        meter_spacegroup.update(spacegroup_mae.item())
        
        # Fractional coordinates
        coord = torch.cat((torch.movedim(crys_tens[1:3+1, skip:, :], 0, 1), 
                           crys_tens[skip:, 1:3+1, :]),
                          dim=2)
        coord_avg = coord.mean(dim=2)
        
        # (Fractional) distance vectors
        dxyz = crys_tens[skip:, skip:, 1:]
        dxyz_avg = (dxyz - torch.transpose(dxyz, 0, 1)) / 2   # Average the reflections (reflections have opposite sign)
        coord_from_graph = (coord_avg.unsqueeze(1).repeat(1, param['max_n_atoms'], 1) - 
                            dxyz_avg)   # Coordinates reconstructed from directional graph
        coord_from_graph_avg = coord_from_graph.mean(dim=0)   # Final coordinates
        coord_mae = l1loss(coord_from_graph,   # MAE of coordinates in graph relative to averaged values
                           coord_from_graph_avg.unsqueeze(0).repeat(param['max_n_atoms'], 1, 1))
        meter_coord.update(coord_mae.item())
        
        # Pairwise distances
        dist = crys_tens[skip:, skip:, 0]
        dist_recon = (coord_from_graph_avg.unsqueeze(1).repeat(1, param['max_n_atoms'], 1) - 
                      coord_from_graph_avg.unsqueeze(0).repeat(param['max_n_atoms'], 1, 1))
        dist_recon = torch.norm(dist_recon, dim=2)
        dist_mae = l1loss(dist, dist_recon)   # MAE of generated pairwise distances and final pairwise distances
        meter_dist.update(dist_mae.item())
        
        # Construct structure
        struc = Atoms(symbols=atom_el_gt, 
                      scaled_positions=coord_from_graph_avg.numpy(),
                      cell=cell,
                      pbc=True)
        struc_all.append(struc)
        
    # Save structures and statistics
    write(write_fname, struc_all, format='extxyz')
    with open('statistics.csv', "w", newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['atom_num_accuracy', 'lat_len_mae', 'lat_angle_mae', 
                            'spacegroup_mae', 'coord_mae', 'pair_dist_mae'])
        csvwriter.writerow([meter_atom_num.avg, meter_lat_len.avg, meter_lat_angle.avg, 
                            meter_spacegroup.avg, meter_coord.avg, meter_dist.avg])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crys_tens_data', type=str, default='crys_tens_gen.pt', help='path to CrysTens (pt file)')
    parser.add_argument('--train_data_dir', type=str, default='/path/to/data/train_val_data/crystens/int', help='path to directory containing CrysTens train/val datasets and json files')
    parser.add_argument('--write_fname', type=str, default='gen.extxyz', help='filename to write extxyz file')
    
    args = parser.parse_args()
    
    with open(args.train_data_dir + '/param.json') as f:
        param = json.load(f)
    with open(args.train_data_dir + '/scaler.json') as f:
        scaler = json.load(f)
    crys_tens_ref = torch.load(args.train_data_dir + '/crys_tens_val.pt')
    skip = 11 + param['padding']   # Number of lines to skip
    atom_num_gt = torch.clone(crys_tens_ref[0, 0, 0, skip:])   # Ground truth atomic number
    atom_num_gt = unnormalize_array(atom_num_gt,
                                    scaler['atom_num']['min'],
                                    scaler['atom_num']['max'])
    atom_num_gt = atom_num_gt.int()
    
    print('Loading CrysTens...')
    stacked_crys_tens = torch.load(args.crys_tens_data)
    print('Unnormalizing CrysTens...')
    stacked_crys_tens = unnormalize_stacked_crys_tens(stacked_crys_tens, param, scaler)
    print('Converting to extxyz...')
    convert_extxyz(stacked_crys_tens, param, atom_num_gt, args.write_fname)


if __name__ == '__main__':
    main()



