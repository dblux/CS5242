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

def create_mlp(dim):
    im_input = Input(shape=dim)
    h = Flatten()(im_input)
    h = Dense(200)(h)
    h = Dropout(0.4)(h)
    h = Activation('relu')(h)
    h = Dense(200)(h)
    h = Activation('relu')(h)
    h = Dense(96)(h)
    h = Activation('relu')(h)
    h = Dense(96)(h)
    h = Activation('relu')(h)
    h = Dense(48)(h)
    h = Activation('relu')(h)
    h = Dense(48)(h)
    h = Activation('relu')(h)
    h = Dense(24)(h)
    h = Activation('relu')(h)
    h = Dense(24)(h)
    h = Activation('relu')(h)
    h = Dense(16)(h)
    h = Activation('relu')(h)
    h = Dense(16)(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(2)(h)
    h = BatchNormalization()(h)
    output = Activation('softmax')(h)
    model = Model(inputs=im_input, outputs=output)
    
    return model

#%%

# Load single input training data

print('Loading X_train...\n')
num_pairs = 3000
pos_dir = '../data/16_8/positive_x/*'
neg_dir = '../data/16_8/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X_train = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_train = X_train[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X_train = np.concatenate((X_train,X_neg), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X_train = np.concatenate((X_train,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
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
                   validation_split=0.2,
                   verbose=1,
                   batch_size=batch_size)

with open('../fig/mlp_history.pkl', 'wb') as file:
    pickle.dump(mlp_hist.history, file)

# ## 学習結果の確認
print('MLP Validation Loss:', mlp_hist.history['val_loss'][-1])
print('MLP Validation Accuracy:', mlp_hist.history['val_categorical_accuracy'][-1])

#%%

# Load testing data

print('Loading X_test...\n')
test_dir = '../data/16_8/test_x/'
test_files_path = sorted(os.listdir(test_dir))
num_pairs = len(test_files_path)
print('X_test:', num_pairs)

# Load first sample
X_test = np.loadtxt(test_dir + test_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_test = X_test[np.newaxis,:,:,np.newaxis]

for i in range(1,824):
    print('X_test sample:', i)
    X_1 = np.loadtxt(test_dir + test_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_1 = X_1[np.newaxis,:,:,np.newaxis]
    X_test = np.concatenate((X_test,X_1), axis=0)

#%%

# Create df to store results

pro_id = np.repeat(range(1,825),824)
lig_id = np.tile(range(1,825),824)
pair_id = pd.DataFrame({'pro_id':pro_id, 'lig_id':lig_id}, columns=['pro_id','lig_id'])
print(pair_id)

predicted_mlp = mlp.predict(X_test, batch_size)
prob = pd.DataFrame(predicted_mlp, columns=['dock','no_dock'])
print(prob)

predict = pd.concat([pair_id, prob], axis=1)
predict.to_csv('../fig/predict.csv', index=False)                                                                                                                                                                         
print('Predict:\n', predict)

test_predictions = np.arange(1,12).reshape(1,-1)
for i in range(1,825):
    top_10 = predict[predict.pro_id==i].sort_values(by='dock', ascending=False)[:10]
    lig_row = np.concatenate((np.array([i]), top_10.lig_id.values))[np.newaxis,:]
    test_predictions = np.concatenate((test_predictions,lig_row), axis=0)
    
lig_predict = pd.DataFrame(test_predictions[1:],
                           columns=['pro_id','lig1_id','lig2_id','lig3_id','lig4_id','lig5_id',
                                     'lig6_id','lig7_id','lig8_id','lig9_id','lig10_id'])
    
lig_predict.to_csv('../fig/test_predictions.txt', sep='\t', index=False)