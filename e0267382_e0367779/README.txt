Predicting Protein – Ligand Interaction by using Deep Learning Models 


Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Directory Structure
```
scripts/
	read_pdb_file.py
	read_testing_pdb_file.py
	train_preprocessing_binary.py
	test_preprocessing_binary.py
  	pos_neg_split.py
 	cnn_binary.py
 	prediction_analysis.py

data/
	training_data/
	testing_data/
	train_pro/
	train_lig/
	test_pro/
	test_lig/
	16_8/

fig

```

*** Copy all scripts to ./scripts/ and place training_data and testing_data in ./data/

Creating Training Dataset
--------------------
1. Run read_pdb_file.py on ../data/training_data to obtain all protein and ligand csv files in ./data/train_pro/
	python read_pdb_file.py

2. Separate protein and ligand csv files into separate directories at ./data/train_pro and ./data/train_lig
	mv ./data/train_pro/*lig* ./data/train_lig

3. Make new directory ../data/train_x
	mkdir ../data/train_x

4. Run train_preprocessing_binary.py on the above csv files. Output will be saved to ../data/train_x
	python train_preprocessing_binary.py

5. Make new directory ../data/positive_x
	mkdir ../data/positive_x

6. Run pos_neg_split.py to separate positive samples into ../data/positive_x
	python pos_neg_split.py

7. Rename ../data/train_x as ../data/negative_x
	mv ../data/train_x ../data/negative_x

8. Make new directory ../data/16_8/ and shift folders ../data/negative_x and ../data/positive_x inside
	mkdir ../data/16_8/
	mv ../data/negative_x ../data/16_8/
	mv ../data/positive_x ../data/16_8/

Creating Test Dataset
--------------------
1. Run read_testing_pdb_file.py on ../data/testing_data to obtain all protein and ligand csv files in ./data/test_pro/
	python read_testing_pdb_file.py

2. Separate protein and ligand csv files into separate directories at ./data/test_pro and ./data/test_lig
	mv ./data/test_pro/*lig* ./data/test_lig

3. Make new directory ../data/test_x
	mkdir ../data/train_x

4. Run test_preprocessing_binary.py on the above csv files. Output will be saved to ../data/test_x
	python train_preprocessing_binary.py

5. Shift folders ../data/test_x inside ../data/16_8/
	mv ../data/test_x ../data/16_8/

Training Model and Predicting
--------------------
1. Run cnn_binary.py , training information and prediction output will be saved in ../fig/
	python cnn_binary.py
	
2. Run prediction_analysis.py to analyse the training information and prediction output