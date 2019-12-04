## Algorithm
- Ligands and proteins have a different number of atoms
- Preserve all atoms in ligands. Make up to n number of atoms for input of one protein and one ligand.
- Correct protein-ligand pairs are properly aligned
- Devise an algorithm to identify the docking site
- Standardise the input of one protein and one ligand into NN

## Neural Network
- Output: Binary classification of Dock vs No Dock. Labels are {0,1}.
- Loss function: Cross entropy loss?
- Architecture: Convolution? How to preserve 3D spatial information?
- 3D Convolution?

## Technical details
1. How to format the inputs to the neural network?
3. Loop through the combinations of right and wrong answers

!!! Remember to set numpy random seed!
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

Categorical inputs: Binary input. one hot or not. embedding.

How to generate and format the dataset - training or labels.

Input:
x y z type (0/1)

Task: Binary classification. Two output nodes (one-hot encoding) with final softmax activation function.

Things to decide on:
- number n (% of protein to include)
- number of negative examples to generate. what is the maximum
- the ligands can have 4 or 5 atoms. is there a way to standardise their size?
- the ligand atoms are relatively little as compared to the protein atoms. is there a way to make sure that they are weighted more?

Order of the training data has to be randomised