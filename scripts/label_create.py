#!/usr/bin/env python3
# Split train_x into positive_x and negative_x

import glob, os

#%%

for i in range(1,11):
    fname = '{:04d}_{:04d}.csv'.format(i,i)
    src = '../data/train_x/' + fname
    dest = '../data/positive_x/' + fname
    print(src, dest)
#    os.rename(src, dest)

#%%


    
