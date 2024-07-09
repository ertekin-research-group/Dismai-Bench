import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='eval_gen.pt', help='path to generated data (pt file)')
    parser.add_argument('--write_fname', type=str, default='gen.extxyz', help='filename to write extxyz file')
    
    args = parser.parse_args()
    
    data = torch.load(args.data, map_location=torch.device("cpu") if not torch.cuda.is_available() else None)
    data['num_atoms'] = torch.reshape(data['num_atoms'], (-1,))
    n_structures = data['num_atoms'].size(dim=0)
    
    data['frac_coords'] = torch.reshape(data['frac_coords'], (-1, 3))
    data['atom_types'] = torch.reshape(data['atom_types'], (-1,))
    data['lengths'] = torch.reshape(data['lengths'], (-1, 3))
    data['angles'] = torch.reshape(data['angles'], (-1, 3))
    ase_adaptor = AseAtomsAdaptor()
    struc_all = []
    atoms_count = 0
    for i in tqdm(range(n_structures)):
        n_atoms = data['num_atoms'][i].item()
        frac_coords = data['frac_coords'][atoms_count:(atoms_count+n_atoms)].numpy()
        atom_types = data['atom_types'][atoms_count:(atoms_count+n_atoms)].numpy()
        lengths = data['lengths'][i].numpy()
        angles = data['angles'][i].numpy()
        atoms_count += n_atoms
        
        lat = Lattice.from_parameters(a=lengths[0], b=lengths[1], c=lengths[2], 
                                      alpha=angles[0], beta=angles[1], gamma=angles[2])
        struc = Structure(lat, atom_types, frac_coords)
        struc = struc.get_sorted_structure()
        struc_all.append(ase_adaptor.get_atoms(struc))
    
    write(args.write_fname, struc_all, format='extxyz') 


if __name__ == '__main__':
    main()


