#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#%%

atoms_lig = pd.read_csv('atoms_lig.csv', sep='\s', header=None, )
atoms_pro = pd.read_csv('atoms_pro.csv', sep='\s', header=None)

atoms_lig = atoms_lig[:-1]
atoms_pro = atoms_pro[:-1]
print(atoms_pro.shape)

median_lig = np.median(atoms_lig[0])
median_pro = np.median(atoms_pro[0])
stats.mode(atoms_pro[0])

plt.figure()
plt.minorticks_on()
plt.xlabel('No. of atoms')
plt.ylabel('Frequency')
plt.hist(atoms_lig[0], np.max(atoms_lig[0])-1)

plt.figure()
plt.minorticks_on()
plt.xlabel('No. of atoms')
plt.ylabel('Frequency')
plt.hist(atoms_pro[0], 100)

#%%
# Proof of knn

from sklearn.neighbors import NearestNeighbors

def centroid(arr):
    center = np.mean(arr, axis=0) 
    return center

np.random.seed(0)

x = np.random.rand(50,1)
y = np.random.rand(50,1)

xy = np.concatenate((x,y), axis=1)

center = centroid(xy)

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nbrs.fit(xy)
knn = nbrs.kneighbors(center.reshape(1,-1), return_distance=False)
xy1 = xy[knn.flatten()]

plt.figure()
plt.scatter(x,y)
plt.scatter(center[0], center[1], c='red')
plt.scatter(xy1[:,0], xy1[:,1], c='orange')