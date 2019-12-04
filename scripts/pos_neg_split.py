#!/usr/bin/env python3
# Extract positive_x from train_x

import os

src_dir = '../data/train_x/'
dest_dir = '../data/positive_x/'

for i in range(1,3001):
    fname = '{:04d}_{:04d}.npy'.format(i,i)
    src = src_dir + fname
    dest = dest_dir + fname
    print(src, dest)
    os.rename(src, dest)