from tqdm import tqdm
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import csv
import argparse
from distutils.util import strtobool
from math import ceil
from matminer.featurizers.base import BaseFeaturizer
from typing import Literal
import copy
import os
import ruamel.yaml as yaml
import pymatgen.analysis.local_env
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams

def load_cn_motif_op_params():
    """
    Load the file for the local env motif parameters into a dictionary.

    Returns:
        (dict)
    """
    with open(
        os.path.join(os.path.dirname(pymatgen.analysis.local_env.__file__), "cn_opt_params.yaml"),
    ) as f:
        return yaml.safe_load(f)

def load_cn_target_motif_op():
    """
    Load the file fpor the

    Returns:
        (dict)
    """
    with open(os.path.join(os.path.dirname(__file__), "cn_target_motif_op.yaml")) as f:
        return yaml.safe_load(f)

class CrystalNNFingerprint_NNSpecies(BaseFeaturizer):
    """
    A local order parameter fingerprint for periodic crystals.

    The fingerprint represents the value of various order parameters for the
    site. The "wt" order parameter describes how consistent a site is with a
    certain coordination number. The remaining order parameters are computed
    by multiplying the "wt" for that coordination number with the OP value.

    The chem_info parameter can be used to also get chemical descriptors that
    describe differences in some chemical parameter (e.g., electronegativity)
    between the central site and the site neighbors.
    
    The output_nn_species toggle allows the output of the anion species 
    (Cl & O) of neighbors
    """

    @staticmethod
    def from_preset(preset: Literal["cn", "ops"], **kwargs):
        """
        Use preset parameters to get the fingerprint
        Args:
            preset ('cn' | 'ops'): Initializes the featurizer to use coordination number ('cn') or structural
                order parameters like octahedral, tetrahedral ('ops').
            **kwargs: other settings to be passed into CrystalNN class
        """
        if preset == "cn":
            op_types = {k + 1: ["wt"] for k in range(24)}
            return CrystalNNFingerprint_NNSpecies(op_types, **kwargs)

        elif preset == "ops":
            cn_target_motif_op = load_cn_target_motif_op()
            op_types = copy.deepcopy(cn_target_motif_op)
            for k in range(24):
                if k + 1 in op_types:
                    op_types[k + 1].insert(0, "wt")
                else:
                    op_types[k + 1] = ["wt"]

            return CrystalNNFingerprint_NNSpecies(op_types, chem_info=None, **kwargs)

        else:
            raise RuntimeError('preset "{}" is not supported in ' "CrystalNNFingerprint_NNSpecies".format(preset))

    def __init__(self, op_types, chem_info=None, output_nn_species=True, **kwargs):
        """
        Initialize the CrystalNNFingerprint_NNSpecies. Use the from_preset() function to
        use default params.
        Args:
            op_types (dict): a dict of coordination number (int) to a list of str
                representing the order parameter types
            chem_info (dict): a dict of chemical properties (e.g., atomic mass)
                to dictionaries that map an element to a value
                (e.g., chem_info["Pauling scale"]["O"] = 3.44)
            output_nn_species (Boolean): When true, outputs the anion species of 
                nearest neighbors (Cl or O) as a fraction of the number of neighbors
            **kwargs: other settings to be passed into CrystalNN class
        """
        
        self.output_nn_species = output_nn_species
        
        self.op_types = copy.deepcopy(op_types)
        self.cnn = CrystalNN(**kwargs)
        if chem_info is not None:
            self.chem_info = copy.deepcopy(chem_info)
            self.chem_props = list(chem_info.keys())
        else:
            self.chem_info = None
        cn_motif_op_params = load_cn_motif_op_params()

        self.ops = {}  # load order parameter objects & paramaters
        for cn, t_list in self.op_types.items():
            self.ops[cn] = []
            for t in t_list:
                if t == "wt":
                    self.ops[cn].append(t)
                else:
                    ot = t
                    p = None
                    if cn in cn_motif_op_params.keys():
                        if t in cn_motif_op_params[cn].keys():
                            ot = cn_motif_op_params[cn][t][0]
                            if len(cn_motif_op_params[cn][t]) > 1:
                                p = cn_motif_op_params[cn][t][1]
                    self.ops[cn].append(LocalStructOrderParams([ot], parameters=[p]))

    def featurize(self, struct, idx):
        """
        Get crystal fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            list of weighted order parameters of target site.
        """

        nndata = self.cnn.get_nn_data(struct, idx)
        max_cn = sorted(self.op_types)[-1]

        cn_fingerprint = []

        if self.output_nn_species == True:
            nn_species_frac_all = []   # Stores the fraction of neighbors that are Cl or O

        if self.chem_info is not None:
            prop_delta = {}  # dictionary of chemical property to final value
            for prop in self.chem_props:
                prop_delta[prop] = 0
            sum_wt = 0
            elem_central = struct.sites[idx].specie.symbol
            specie_central = str(struct.sites[idx].specie)

        for k in range(max_cn):
            cn = k + 1
            wt = nndata.cn_weights.get(cn, 0)
            if cn in self.ops:
                for op in self.ops[cn]:
                    if op == "wt":
                        cn_fingerprint.append(wt)
                        
                        if self.output_nn_species == True :
                            if wt != 0:
                                neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]
                                Cl_count = 0
                                O_count = 0
                                for neigh in neigh_sites:
                                    elem_neigh = neigh.specie.symbol
                                    if elem_neigh == 'Cl':
                                        Cl_count += 1
                                    if elem_neigh == 'O':
                                        O_count += 1
                                #if Cl_count + O_count != cn:
                                #    raise Exception("Neighbor species count doesn't add up to coordination number "+str(cn)+" for atom index "+str(idx))
                                nn_species_frac_all.append(Cl_count/cn)
                                nn_species_frac_all.append(O_count/cn)
                            else:
                                nn_species_frac_all.append(0)
                                nn_species_frac_all.append(0)

                        if self.chem_info is not None and wt != 0:
                            # Compute additional chemistry-related features
                            sum_wt += wt
                            neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]

                            for prop in self.chem_props:
                                # get the value for specie, if not fall back to
                                # value defined for element
                                prop_central = self.chem_info[prop].get(
                                    specie_central,
                                    self.chem_info[prop].get(elem_central),
                                )

                                for neigh in neigh_sites:
                                    elem_neigh = neigh.specie.symbol
                                    specie_neigh = str(neigh.specie)
                                    prop_neigh = self.chem_info[prop].get(
                                        specie_neigh,
                                        self.chem_info[prop].get(elem_neigh),
                                    )

                                    prop_delta[prop] += wt * (prop_neigh - prop_central) / cn

                    elif wt == 0:
                        cn_fingerprint.append(wt)
                    else:
                        neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]
                        opval = op.get_order_parameters(
                            [struct[idx]] + neigh_sites,
                            0,
                            indices_neighs=[i for i in range(1, len(neigh_sites) + 1)],
                        )[0]
                        opval = opval or 0  # handles None
                        cn_fingerprint.append(wt * opval)
        chem_fingerprint = []

        if self.chem_info is not None:
            for val in prop_delta.values():
                chem_fingerprint.append(val / sum_wt)

        return cn_fingerprint + chem_fingerprint + nn_species_frac_all

    def feature_labels(self):
        labels = []
        max_cn = sorted(self.op_types)[-1]
        for k in range(max_cn):
            cn = k + 1
            if cn in list(self.ops.keys()):
                for op in self.op_types[cn]:
                    labels.append(f"{op} CN_{cn}")            
        if self.chem_info is not None:
            for prop in self.chem_props:
                labels.append(f"{prop} local diff")
        if self.output_nn_species == True:
            for k in range(max_cn):
                cn = k + 1
                if cn in list(self.ops.keys()):
                    labels.append(f"Cl fraction coord num {cn}")
                    labels.append(f"O fraction coord num {cn}")         
        return labels

    def citations(self):
        return []

    def implementors(self):
        return ["Anubhav Jain", "Nils E.R. Zimmermann", "Adrian Xiao Bin Yong"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split run into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_relaxed.extxyz', help='path to data (extxyz file)')
    
    args = parser.parse_args()
    batch = args.batch
    
    if args.split_batches:
        data = read(filename='../'+args.data_path, index=':', format='extxyz')
    else:
        data = read(filename=args.data_path, index=':', format='extxyz')
    csv_fname_Li = 'cnn_stats_Li.csv'
    csv_fname_Sc = 'cnn_stats_Sc.csv'
    csv_fname_Co = 'cnn_stats_Co.csv'
    
    # Get starting and ending indexes of structures to analyze
    if args.split_batches:
        n_strucs_per_batch = args.n_strucs_per_batch
        i_start = (batch - 1)*n_strucs_per_batch   # Starting index
        if batch == ceil(len(data) / n_strucs_per_batch):   # last batch may have less than n_strucs_per_batch
            i_end = len(data) - 1   # Ending index
        else:
            i_end = batch*n_strucs_per_batch - 1   # Ending index
    else:
        i_start = 0
        i_end = len(data) - 1 
    
    # Define order parameters to analyze
    op_types_dict = {1: ['wt', 'sgl_bd'],
                     2: ['wt',
                      'L-shaped',
                      'water-like',
                      'bent 120 degrees',
                      'bent 150 degrees',
                      'linear'],
                     3: ['wt', 'trigonal planar', 'trigonal non-coplanar', 'T-shaped'],
                     4: ['wt',
                      'square co-planar',
                      'tetrahedral',
                      'rectangular see-saw-like',
                      'see-saw-like',
                      'trigonal pyramidal'],
                     5: ['wt', 'pentagonal planar', 'square pyramidal', 'trigonal bipyramidal'],
                     6: ['wt', 'hexagonal planar', 'octahedral', 'pentagonal pyramidal'],
                     7: ['wt', 'hexagonal pyramidal', 'pentagonal bipyramidal'],
                     8: ['wt', 'body-centered cubic', 'hexagonal bipyramidal'],
                     9: ['wt', 'q2', 'q4', 'q6'],
                     10: ['wt', 'q2', 'q4', 'q6'],
                     11: ['wt', 'q2', 'q4', 'q6'],
                     12: ['wt', 'cuboctahedral', 'q2', 'q4', 'q6'],
                     13: ['wt'],
                     14: ['wt'],
                     15: ['wt'],
                     16: ['wt'],
                     17: ['wt'],
                     18: ['wt'],
                     19: ['wt'],
                     20: ['wt'],
                     21: ['wt'],
                     22: ['wt'],
                     23: ['wt'],
                     24: ['wt']
                     }
    cnn = CrystalNNFingerprint_NNSpecies(op_types=op_types_dict, cation_anion=True, output_nn_species=True)
    
    # Write csv header
    for csv_fname in [csv_fname_Li, csv_fname_Sc, csv_fname_Co]:
        with open(csv_fname, "w", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['structure_number', 'atom_number'] + cnn.feature_labels())
    
    # Analyze coordination environment
    print('Analyzing coordination environment')
    ase_adaptor = AseAtomsAdaptor()
    for i in tqdm(range(i_start, i_end+1)):
        atoms = data[i]
        struc = ase_adaptor.get_structure(atoms)
        # Add oxidation state to allow CrystalNN to identify which are cations and anions
        struc.add_oxidation_state_by_element({'Li':1, 'Sc':3, 'Co':3, 'Cl':-1, 'O':-2})  
        
        indices_Li = [idx for idx, site in enumerate(struc) if site.species_string == 'Li+']
        indices_Sc = [idx for idx, site in enumerate(struc) if site.species_string == 'Sc3+']
        indices_Co = [idx for idx, site in enumerate(struc) if site.species_string == 'Co3+']
        for el in [(indices_Li, csv_fname_Li), (indices_Sc, csv_fname_Sc), (indices_Co, csv_fname_Co)]:
            indices_el = el[0]
            csv_fname = el[1]
            for j in indices_el:
                stat = cnn.featurize(struc, j)
                with open(csv_fname, "a", newline='') as csvfile: 
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([i+1, j+1] + stat)    
    
    ## Get the most likely coordination number and order parameter
    for csv_fname in [csv_fname_Li, csv_fname_Sc, csv_fname_Co]:
        print('Processing output file `{}`'.format(csv_fname))
        df = pd.read_csv(csv_fname)
        
        # Find the most likely coordination number
        df_cn = df.filter(like='wt CN')
        df['most likely CN'] = df_cn.idxmax(axis=1).str.split('_').str[-1].astype(int)
        df['CN likelihood'] = df_cn.max(axis=1)
        
        # Find the anion counts and the most likely order parameter
        count_Cl_all = []
        count_O_all = []
        OP_type_all = []
        OP_val_all = []
        for i in tqdm(range(len(df))):
            cn = df['most likely CN'][i]
            count_Cl = round(df['Cl fraction coord num '+str(cn)][i] * cn)
            count_O = round(df['O fraction coord num '+str(cn)][i] * cn)
            count_Cl_all.append(count_Cl)
            count_O_all.append(count_O)
            df_op = df.iloc[i].filter(like='CN_'+str(cn))
            df_op = df_op.drop(labels=['wt CN_'+str(cn)])
            if len(df_op) > 0:
                OP_type_all.append(df_op.idxmax(axis=0))
                OP_val_all.append(df_op.max(axis=0))
            else:
                OP_type_all.append('None CN_'+str(cn))
                OP_val_all.append(None)
        df['Cl count (most likely CN)'] = pd.Series(count_Cl_all)
        df['O count (most likely CN)'] = pd.Series(count_O_all)
        df['most likely OP'] = pd.Series(OP_type_all)
        df['OP value'] = pd.Series(OP_val_all)
        
        # Rearrange new columns to the front
        cols = list(df.columns)
        cols = cols[:2] + cols[-6:] + cols[2:-6]
        df = df[cols]
        df.to_csv(csv_fname, index=False)


if  __name__ == '__main__':
    main()

