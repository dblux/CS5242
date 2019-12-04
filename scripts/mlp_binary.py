#!/usr/bin/env python3
#%%

import os
import glob
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential, Model        
from keras.layers import Input, Dense, Conv2D, Activation, Dropout, Flatten, Concatenate, BatchNormalization

#%%

def create_cnn(dim=(24,4,1)):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(8,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
   
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model

#%%

# Load single input training data

print('Loading X_train...\n')
num_pairs = 3000
pos_dir = '../data/16_8_3/positive_x/*'
neg_dir = '../data/16_8_3/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X_train = np.load(pos_files_path[0])
X_train = X_train[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.load(neg_files_path[0])
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X_train = np.concatenate((X_train,X_neg), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.load(pos_files_path[i])
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X_train = np.concatenate((X_train,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.load(neg_files_path[i])
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X_train = np.concatenate((X_train,X_neg), axis=0)

print('X_train.shape:', X_train.shape)

# Create Y: Dock-(1,0), No dock-(0,1)
Y_train = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y_train.shape:', Y_train.shape)

#%%

batch_size = 200
n_epoch = 150

mlp = create_mlp(dim=(24,4,1))
print(mlp.summary())

mlp.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
mlp_hist = mlp.fit(X_train, Y_train,
                   epochs=n_epoch,
                   verbose=1,
                   batch_size=batch_size)

with open('../fig/mlp_history.pkl', 'wb') as file:
    pickle.dump(mlp_hist.history, file)

# ## 学習結果の確認
print('MLP Validation Loss:', mlp_hist.history['loss'][-1])
print('MLP Validation Accuracy:', mlp_hist.history['categorical_accuracy'][-1])

#%%

# Load testing data

print('Loading X_test...\n')
test_dir = '../data/16_8_3/test_x/'
test_files_path = sorted(os.listdir(test_dir))
num_pairs = len(test_files_path)
print('X_test:', num_pairs)

#%%

# Load and evaluate test data

id_arr = np.arange(1,11)[np.newaxis,:]
pro_prob = np.arange(1,825)[np.newaxis,:]

for i in range(824):
    # Load first protein
    k = i*824
    print(test_files_path[k])
    X_test = np.load(test_dir + test_files_path[k])
    X_test = X_test[np.newaxis,:,:,np.newaxis]
    
    for j in range(1,824):
        X_1 = np.load(test_dir + test_files_path[k+j])
        print(test_files_path[k+j])
        X_1 = X_1[np.newaxis,:,:,np.newaxis]
        X_test = np.concatenate((X_test,X_1), axis=0)
    
    predicted_mlp = mlp.predict(X_test, batch_size)
    
    pro_row = predicted_mlp[:,0][np.newaxis,:]
    pro_prob = np.concatenate((pro_prob, pro_row), axis=0)
    
    lig_id = predicted_mlp[:,0].argsort()[-10:][::-1] + 1
    lig_id = lig_id[np.newaxis,:]
    id_arr = np.concatenate((id_arr, lig_id), axis=0)
    
    print('Protein: ', i + 1)

np.save('../fig/lig_predictions.npy', id_arr)
np.save('../fig/pro_prob.npy', pro_prob)
print('Ligands predicted!')