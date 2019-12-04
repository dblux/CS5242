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

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))
        
        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append(1.) # 'h'/1 means hydrophobic
        else:
            atomtype_list.append(-1.) # 'p'/-1 means polar
            
    mat = np.column_stack((X_list, Y_list, Z_list, atomtype_list))
    
    write_dir = "../data/train_1_pro/"
    fname = write_dir + filename.split("/")[-1].split(".")[0] + ".csv"
    print(fname)
    np.savetxt(fname, mat, delimiter=",")
    
#%%
import numpy as np, glob    

read_pdb("/Users/weixin/data/cs5242/training_data/0001_lig_cg.pdb")

#%%
import numpy as np, glob

read_dir = "../data/training_data/*"
files = glob.glob(read_dir)

for i in files:
    read_pdb(i)