# Count number of lines in file and sort
wc -l train_lig/*.csv | sort -k1 -n > atoms_lig.csv

# scp
scp foo.pdf weixin@xgpd3:~/project


# Initialise conda
source .bashrc
conda activate