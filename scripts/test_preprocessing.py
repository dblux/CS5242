#!/usr/bin/env python3
#%%

import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)

def centroid(arr):
    center = np.mean(arr, axis=0) 
    return center

#%%

lig_dir = '../data/test_lig/*'
pro_dir = '../data/test_pro/*'
output_dir = '../data/test_x/'

num_atoms_lig = 8
num_atoms_pro = 16

lig_files_path = glob.glob(lig_dir)
pro_files_path = glob.glob(pro_dir)

lig_files_path.sort()
pro_files_path.sort()

nbrs_lig = NearestNeighbors(n_neighbors=num_atoms_lig, algorithm='ball_tree')
nbrs_pro = NearestNeighbors(n_neighbors=num_atoms_pro, algorithm='ball_tree')

for lig_file in lig_files_path:
    lig_arr = np.loadtxt(lig_file, delimiter=',', dtype=np.float64, ndmin=2)
    lig_num = lig_file[-15:-11]
    lig_index = int(lig_num) - 1
    print('Ligand no.:', lig_num)
    print('Ligand array shape: {}'.format(lig_arr.shape))

    # Calculate centroid
    lig_arr1 = np.copy(lig_arr[:,:-1])
    lig_centroid = centroid(lig_arr1)
    print('Centroid:\n {}'.format(lig_centroid))
    
    lig_atoms = lig_arr.shape[0]
    if lig_atoms < num_atoms_lig:
        # Pad with zeros at the bottom
        lig_arr_fixed = np.pad(lig_arr, ((0,num_atoms_lig - lig_atoms),(0,0)), 'constant')
        print('Padded:\n{}\n'.format(lig_arr_fixed))
    elif lig_atoms > num_atoms_lig:
        # Truncate n number of atoms closest to centroid
        # Ordered with atom nearest to centroid on top
        nbrs_lig.fit(lig_arr1)
        knn = nbrs_lig.kneighbors(lig_centroid.reshape(1,-1), return_distance=False)
        lig_arr_fixed = lig_arr[knn.flatten()]
        print('Truncated:\n{}\n'.format(lig_arr_fixed))
    else:
        lig_arr_fixed = lig_arr
        print('No padding:\n{}\n'.format(lig_arr_fixed))
        
    for pro_file in pro_files_path:
        pro_arr = np.loadtxt(pro_file, delimiter=',', dtype=np.float64, ndmin=2)
        pro_arr1 = np.copy(pro_arr[:,:-1])
        pro_num = pro_file[-15:-11]
        print('Protein no.:', pro_num)
        print('Protein array shape: {}'.format(pro_arr.shape))
        
        pro_atoms = pro_arr.shape[0]
        if pro_atoms < num_atoms_pro:
            # Pad with zeros at the top
            pro_arr_fixed = np.pad(pro_arr, ((num_atoms_pro - pro_atoms,0),(0,0)), 'constant')
            print('Padded: {}'.format(pro_arr_fixed.shape))
        elif pro_atoms > num_atoms_pro:
            # Select n number of protein atoms closest to ligand centroid
            # Ordered with atom nearest to centroid on the bottom
            nbrs_pro.fit(pro_arr1)
            knn = nbrs_pro.kneighbors(lig_centroid.reshape(1,-1), return_distance=False)
            pro_arr_fixed = pro_arr[knn.flatten()][::-1]
            print('Truncated: {}'.format(pro_arr_fixed.shape))
        else:
            pro_arr_fixed = pro_arr
            print('No padding: {}'.format(pro_arr_fixed.shape))
            
        # Concatenate and save
        pro_lig = np.concatenate((pro_arr_fixed, lig_arr_fixed), axis=0)
        pro_lig_fname = output_dir + pro_num + '_' + lig_num + '.csv'
        np.savetxt(pro_lig_fname, pro_lig, delimiter=",")
        print(pro_lig_fname, pro_lig.shape)
        print(pro_lig, '\n')