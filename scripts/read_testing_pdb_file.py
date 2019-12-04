import numpy as np
import glob

def read_pdb(filename):
    
    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()
        # print(stripped_line)

        splitted_line = stripped_line.split('\t')
        
        X_list.append(float(splitted_line[0]))
        Y_list.append(float(splitted_line[1]))
        Z_list.append(float(splitted_line[2]))
        atomtype_list.append(str(splitted_line[3]))
        bool_list = [1 if i == 'h' else -1 for i in atomtype_list]
        
    arr = np.column_stack((X_list, Y_list, Z_list, bool_list))
    
    write_dir = '../data/test_1_pro/'
    fname = write_dir + filename[-15:-4] + '.csv'
    print(fname)
    np.savetxt(fname, arr, delimiter=',')
    
#%%

read_dir = "../data/testing_data/*"
files = glob.glob(read_dir)

for i in files:
    read_pdb(i)