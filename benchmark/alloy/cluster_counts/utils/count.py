import math
import copy
import symop
import numpy as np
from pymatgen.core.sites import PeriodicSite


def apply_basis(str_list):
    """
    apply lattice vector to get Cartesian coordinate for lattice points
    :param str_list: list of structure metadata generated by parse_str function
    :return: str_list with 'LatPnt' in Cartesian coordinate
    """
    cart_str_list = copy.deepcopy(str_list)
    for i in range(len(str_list)):
        str_dict = str_list[i]
        for j in range(len(str_dict['LatPnt'])):
            cart_str_list[i]['LatPnt'][j] = np.dot(np.transpose(str_dict['LatPnt'][j]), str_dict['LatVec'])

    return cart_str_list


def calc_dist(pnt1, pnt2):
    """
    calculate the distance between two point in 3D Cartesian coordinate
    :param pnt1: list
    :param pnt2: list
    :return: dist
    """
    dist = np.linalg.norm(np.array(pnt1) - np.array(pnt2))

    return dist


def scale_clust(clust):
    """
    Transform from unscaled Cartesian coordinates to scaled ones for a given cluster
    :param clust: cluster from clust_list in the format of [[coord_list], [dist_scale], [spin_flag]]
    :return: clusters with scaled coordinates
    """
    size = len(clust[0])
    max_dist = 0
    scaled_clust = copy.deepcopy(clust)
    if clust[1][0] == 1:
        return scaled_clust
    else:
        for i in range(size):
            for j in range(i, size):
                dist = calc_dist(clust[0][i], clust[0][j])
                if max_dist < dist:
                    max_dist = dist
        if max_dist != 0:
            scale = max(clust[1]) / max_dist
        else:
            scale = 0
        for i in range(size):
            for j in range(3):
                scaled_clust[0][i][j] = scale * clust[0][i][j]
        scaled_clust[1][0] = 1
        return scaled_clust


def frac_to_cart(frac_coord, basis):
    """
    Transform from fraction/direct coord to Cartesian coord for a given structure
    :param frac_coord: list in the format of [0.5, 0.5, 0.5]
    :param basis: list in the format of [[0.0, 0.0, 3.6], [0.0, 3.6, 0.0], [3.6, 0.0, 0.0]]
    :return: cart_coord
    """
    frac_coord = np.array(frac_coord)
    trans_matr = np.vstack(basis).T
    cart_coord = np.matmul(trans_matr, frac_coord.T).T

    return list(cart_coord)


def cart_to_frac(cart_coord, basis):
    """
    transform from Cartesian coordinate to direct/fraction coordinate
    :param cart_coord: list in the format of [1.8, 1.8, 1.8]
    :param basis: list in the format of [[0.0, 0.0, 3.6], [0.0, 3.6, 0.0], [3.6, 0.0, 0.0]]
    :return: frac_coord
    """
    cart_coord = np.array(cart_coord)
    trans_matr = np.vstack(basis).T
    inv_matr = np.linalg.inv(trans_matr)
    frac_coord = np.matmul(inv_matr, cart_coord.T).T

    return list(frac_coord)


def apply_pbc(clust, str_dict):
    """
    Apply periodic boundary conditions to the sites of a given cluster
    :param clust: containing scaled Cartesian coordinates
    :param str_dict: containing Cartesian coordinates
    :return: frac_clust
    """
    size = len(clust[0])
    pbc_clust = copy.deepcopy(clust)
    for i in range(size):
        pbc_clust[0][i] = cart_to_frac(clust[0][i], str_dict['LatVec'])
        pbc_clust[0][i] = np.around(pbc_clust[0][i], decimals=3)
        for j in range(3):
            pbc_clust[0][i][j] = pbc_clust[0][i][j] % 1
        pbc_clust[0][i] = frac_to_cart(pbc_clust[0][i], str_dict['LatVec'])

    return pbc_clust


def find_spec(site_idx, clust, clust_ref, struc, str_dict, margin):
    """
    find the species on each sites of a given cluster
    :param site_idx: Index of site the cluster is centered on
    :param clust: cluster after rotation and applying PBC
    :param clust_ref: cluster before rotation (no PBC)
    :param struc: Pymatgen Structure object
    :param str_dict: Dictionary containing information about the structure
    :param margin: margin of error for distance between cluster coordinate and 
                   actual site coordinate (unit: Angstrom) 
    :return: spec_list: a list of species in the same sequence of the cluster sites
    """
    spec = [None] * len(clust[0])
    for i in range(len(clust[0])):
        if clust_ref[0][i] == [0., 0., 0.]:
            # Simply take the species of the site atom
            spec[i] = struc[site_idx].specie.symbol
        else:
            # Find the closest atom to the designated cluster coordinate
            clust_coord = np.array(clust[0][i]) + 1e-06   # Pymatgen's get_neighbor_list() fails if cluster coord contains exactly 0.
            clust_site = PeriodicSite('H', clust_coord, struc.lattice,
                                      coords_are_cartesian=True)
            _ , struc_site_all, _ , dist_all = struc.get_neighbor_list(margin, [clust_site])
            n_sites = len(struc_site_all)
            if n_sites > 0:
                spec[i] = struc[struc_site_all[dist_all.argmin()]].specie.symbol
            #if n_sites == 1:   # there should only be 1 site found within margin of error
            #    spec[i] = struc[struc_site_all[0]].specie.symbol
            #elif n_sites > 1:
            #    print('More than 1 atom found near cluster coordinate for '
            #          'Structure index {}, please check!'.format(str_dict['StrIdx']))
    if None in spec:
        spec_list = ['empty']
    else:
        spec_list = copy.deepcopy(spec)

    return spec_list


def find_spec_and_check_vac(site_idx, clust, clust_ref, struc, str_dict, margin,
                            has_vac):
    """
    find the species on each sites of a given cluster, 
    and counts the number of empty clusters
    :param site_idx: Index of site the cluster is centered on
    :param clust: cluster after rotation and applying PBC
    :param clust_ref: cluster before rotation (no PBC)
    :param struc: Pymatgen Structure object
    :param str_dict: Dictionary containing information about the structure
    :param margin: margin of error for distance between cluster coordinate and 
                   actual site coordinate (unit: Angstrom) 
    :param has_vac: Boolean of whether the structure has vacancy/vacancies
    :return: spec_list: a list of species in the same sequence of the cluster sites
    :return: has_vac: Boolean of whether the structure has vacancy/vacancies
    """
    spec = [None] * len(clust[0])
    for i in range(len(clust[0])):
        if clust_ref[0][i] == [0., 0., 0.]:
            # Simply take the species of the site atom
            spec[i] = struc[site_idx].specie.symbol
        else:
            # Find the closest atom to the designated cluster coordinate
            clust_coord = np.array(clust[0][i]) + 1e-06   # Pymatgen's get_neighbor_list() fails if cluster coord contains exactly 0.
            clust_site = PeriodicSite('H', clust_coord, struc.lattice,
                                      coords_are_cartesian=True)
            _ , struc_site_all, _ , dist_all = struc.get_neighbor_list(margin, [clust_site])
            n_sites = len(struc_site_all)
            if n_sites > 0:
                spec[i] = struc[struc_site_all[dist_all.argmin()]].specie.symbol
            #if n_sites == 1:   # there should only be 1 site found within margin of error
            #    spec[i] = struc[struc_site_all[0]].specie.symbol
            #elif n_sites > 1:
            #    print('More than 1 atom found near cluster coordinate for '
            #          'Structure index {}, please check!'.format(str_dict['StrIdx']))
            elif n_sites == 0:
                has_vac = True
    if None in spec:
        spec_list = ['empty']
    else:
        spec_list = copy.deepcopy(spec)

    return spec_list, has_vac


def find_and_fill_vac(clust, clust_ref, struc_fill, str_dict, margin):
    """
    looks for vacancies and fills in the vacancies with dummy atoms
    :param clust: cluster after rotation and applying PBC
    :param clust_ref: cluster before rotation (no PBC)
    :param struc_fill: Pymatgen Structure object for filling in vacancies
    :param str_dict: Dictionary containing information about the structure
    :param margin: margin of error for distance between cluster coordinate and 
                   actual site coordinate (unit: Angstrom)
    :return: struc_fill: Pymatgen Structure object with filled in vacancies
    :return: fill_count: number of vacancies filled in
    """
    fill_count = 0
    for i in range(len(clust[0])):
        if clust_ref[0][i] == [0., 0., 0.]:
            # Ignore the site atom
            pass
        else:
            # Find the closest atom to the designated cluster coordinate
            clust_coord = np.array(clust[0][i]) + 1e-06   # Pymatgen's get_neighbor_list() fails if cluster coord contains exactly 0.
            clust_site = PeriodicSite('H', clust_coord, struc_fill.lattice,
                                      coords_are_cartesian=True)
            _ , struc_site_all, _ , dist_all = struc_fill.get_neighbor_list(margin, [clust_site])
            n_sites = len(struc_site_all)
            if n_sites == 0:   # Vacancy found
                struc_fill.append('H', clust_coord, coords_are_cartesian=True)
                fill_count += 1
            #elif n_sites > 1:
            #    print('When counting vacancies, more than 1 atom found near cluster coordinate {} '
            #          'for Structure index {}, please check!'.format(clust_coord, str_dict['StrIdx']))

    return struc_fill, fill_count


def find_spin(clust, str_dict):
    """
    find the species on each sites of a given cluster
    :param clust: containing Cartesian coordinate after applying PBC
    :param str_dict: containing Cartesian coordinates
    :return: spin product of all cluster sites
    """
    spin = np.zeros(len(clust[0]))
    for i in range(len(clust[0])):
        dist = 0.2
        for j in range(len(str_dict['LatPnt'])):
            new_dist = np.sum(np.abs(np.subtract(clust[0][i], str_dict['LatPnt'][j])))
            if new_dist < dist:
                dist = new_dist
                spin[i] = str_dict['Spin'][j]
    spin_value = math.prod(spin)

    return spin_value


def count_clusters(symeq_clust_list, symeq_clust_rot_list, pntsym_list, 
                   str_dict, struc, clust_list, margin):
    """
    count the number of each cluster for each structure with single lattice
    :param symeq_clust_list: list of symmetry equivalent clusters
    :param symeq_clust_rot_list: list of rotated symmetry equivalent clusters
    :param pntsym_list: list of point symmetry operation for each symmetry equivalent cluster
    :param str_dict: dictionary of structure
    :param struc: Pymatgen Structure object
    :param clust_list: parsed cluster list
    :param margin: margin of error for distance between cluster coordinate and 
                   actual site coordinate (unit: Angstrom) 
    :return: count_list: counts of clusters
    :return: has_vac: Boolean of whether the structure has vacancy/vacancies
    """
    has_vac = False   # Indicates whether the structure has vacancy/vacancies
    count_list = copy.deepcopy(clust_list)
    for i in range(len(clust_list)):
        count_dict = {}
        orig_clust = clust_list[i]
        symeq_clust = symeq_clust_rot_list[i]
        multiplicity = len(symeq_clust)
        count_list[i].append({'Multiplicity': int(multiplicity/len(orig_clust[0]))})
        for j in range(len(str_dict['LatPnt'])):
            for k in range(multiplicity):
                old_clust = symeq_clust[k]
                new_clust = copy.deepcopy(old_clust)   # centered around str_dict['LatPnt'][j]
                for x in range(len(old_clust[0])):
                    new_clust[0][x] = np.sum([old_clust[0][x], str_dict['LatPnt'][j]], axis=0)
                pbc_clust = apply_pbc(new_clust, str_dict)
                if i == 1 and has_vac == False:   # 1st NN dimers, check for vacancies
                    spec, has_vac = find_spec_and_check_vac(j, pbc_clust, old_clust, struc, 
                                                            str_dict, margin, has_vac)
                else:
                    spec = find_spec(j, pbc_clust, old_clust, struc, str_dict, margin)
                if spec != ['empty']:
                    # find the only true equivalent sequence
                    spec = symop.find_eq_spec_seq(list(spec), symeq_clust_list[i][k],
                                                  pntsym_list[i][k])
                    if pbc_clust[2][0] == 0:  # chem term
                        if str(spec) in count_dict.keys():
                            count_dict[str(spec)] += 1
                        else:
                            count_dict[str(spec)] = 1
                    elif pbc_clust[2][0] == 1:  # spin term
                        spin = find_spin(pbc_clust, str_dict)
                        if str(spec) in count_dict.keys():
                            count_dict[str(spec)] += spin
                        else:
                            count_dict[str(spec)] = spin
        for keys in count_dict:
            values = count_dict[keys]
            count_dict[keys] = np.around(values/(str_dict['AtomSum']*len(orig_clust[0])), decimals=5)
        count_list[i].append(count_dict)        

    return count_list, has_vac


def count_vac(symeq_clust_rot_list, str_dict, struc, margin):
    """
    Counts the number of vacancies in the structure. It does so by going through
    1st nearest neighbor pairs, counting the number of vacancies, filling in 
    those vacancies in a dummy structure, then repeat until all vacancies are filled in.
    :param symeq_clust_rot_list: list of rotated symmetry equivalent clusters
    :param str_dict: dictionary of structure
    :param struc: Pymatgen Structure object
    :param margin: margin of error for distance between cluster coordinate and 
                   actual site coordinate (unit: Angstrom) 
    :return: vac_count: number of vacancies in the structure
    """
    vac_count = 0   # Initialize vacancy count for the structure
    struc_fill = struc.copy()   # Structure for filling in the vacancies
    symeq_clust = symeq_clust_rot_list[1]
    multiplicity = len(symeq_clust)
    fill_count = 999   # Initialize number of vacancies filled in
    while fill_count != 0:
        for j in range(len(str_dict['LatPnt'])):
            for k in range(multiplicity):
                old_clust = symeq_clust[k]
                new_clust = copy.deepcopy(old_clust)   # centered around str_dict['LatPnt'][j]
                for x in range(len(old_clust[0])):
                    new_clust[0][x] = np.sum([old_clust[0][x], str_dict['LatPnt'][j]], axis=0)
                pbc_clust = apply_pbc(new_clust, str_dict)
                struc_fill, fill_count = find_and_fill_vac(pbc_clust, old_clust, struc_fill, 
                                                           str_dict, margin)
                vac_count += fill_count
    
    return vac_count    


def count_spin_pair(symeq_clust_list, pntsym_list, str_list, clust_list):
    """
    count the number of each cluster for each structure with single lattice
    :param symeq_clust_list: list of symmetry operation based on the input lattice file defined like ATAT
    :param pntsym_list: list of point symmetry operation for each symmetry equivalent cluster
    :param str_list: parsed DFT data list
    :param clust_list: parsed cluster list
    :return: list of the count number (count_list)
    """
    str_list = apply_basis(str_list)  # transform str from direct coord to Cartesian coord
    count_list_all = []
    for i in range(len(clust_list)):
        if clust_list[i][2][0] == 1:
            count_dict = {}
            for str_dict in str_list:
                symeq_clust = symeq_clust_list[i]
                multiplicity = len(symeq_clust)
                for j in range(len(str_dict['LatPnt'])):
                    for k in range(multiplicity):
                        old_clust = symeq_clust[k]
                        vect = np.subtract.reduce([str_dict['LatPnt'][j], [0, 0, 0]], axis=0)
                        new_clust = copy.deepcopy(old_clust)
                        for x in range(len(old_clust[0])):
                            new_clust[0][x] = np.sum([old_clust[0][x], vect], axis=0)
                        pbc_clust = apply_pbc(new_clust, str_dict)
                        spec = find_spec(pbc_clust, str_dict)
                        if spec != ['empty']:
                            spec = symop.find_eq_spec_seq(list(spec), old_clust, pntsym_list[i][k])
                            spin = find_spin(pbc_clust, str_dict)
                            if str(spec) in count_dict.keys():
                                count_dict[str(spec)][int(spin+1)] += 1
                            else:
                                count_dict[str(spec)] = [0, 0, 0]
                                count_dict[str(spec)][int(spin + 1)] = 1
            count_list_all.append([i, count_dict])

    return count_list_all
