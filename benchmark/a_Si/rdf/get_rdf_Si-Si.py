from vasppy.rdf import RadialDistributionFunction
import pandas as pd
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='gen_relaxed.extxyz', help='path to data (extxyz file)')
    
    args = parser.parse_args()
    structure_all = read(args.data_path, index=':', format='extxyz')
    csv_fname = 'rdf_Si-Si.csv'
    
    ase_adaptor = AseAtomsAdaptor()
    for i in tqdm(range(len(structure_all))):
        structure = ase_adaptor.get_structure(structure_all[i])
        structure.make_supercell(2)
        indices_Si = [idx for idx, site in enumerate(structure) if site.species_string == 'Si']
        rdf_Si_Si = RadialDistributionFunction(structures=[structure], 
                                               indices_i=indices_Si, indices_j=indices_Si)
        
        df = pd.DataFrame.from_dict({i+1: rdf_Si_Si.rdf}, orient='index', columns=rdf_Si_Si.r)
        df.index.name = 'structure_number'
        if i == 0:
            df.to_csv(csv_fname, index=True)
        else:
            df.to_csv(csv_fname, index=True, mode='a', header=False)


if  __name__ == '__main__':
    main()