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
    h = Dense(1200, activation='relu')(h)
    h = Dense(600, activation='relu')(h)
    h = Dense(300, activation='relu')(h)
    h = Dense(300, activation='relu')(h)
    h = Dense(300, activation='relu')(h)
    output = Dense(2, activation='softmax')(h)
    model = Model(inputs=im_input, outputs=output)
    
    return model

#-----------
# CNNモデル
#-----------
def create_cnn(dim):
    model = Sequential()
    
    model.add(Conv2D(32, input_shape=dim,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model


def create_lstm(time_steps):
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(batch_size, time_steps, 4), stateful=False))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def create_mlp_mult(dim1, dim2):
    pro_input = Input(shape=dim1)
    h1 = Flatten()(pro_input)
    h1 = Dense(200, activation='relu')(h1)
    h1 = Dense(200, activation='relu')(h1)
    h1 = Dense(100, activation='relu')(h1)
    
    lig_input = Input(shape=dim2)
    h2 = Flatten()(lig_input)
    h2 = Dense(200, activation='relu')(h2)
    h2 = Dense(200, activation='relu')(h2)
    h2 = Dense(100, activation='relu')(h2)
    
    h3 = Concatenate(axis=-1)([h1,h2])
    h3 = Dense(200, activation='relu')(h3)
    h3 = Dense(200, activation='relu')(h3)
    h3 = Dense(100, activation='relu')(h3)   
    output = Dense(2, activation='softmax')(h3)
    model = Model(inputs=[pro_input, lig_input], outputs=output)

    return model

def create_multiple_cnn(dim1, dim2):
    pro_input = Input(shape=dim1)
    h1 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(pro_input)
    h1 = Conv2D(32, kernel_size=(5, 3),
                strides=(3, 1),
                padding='valid',
                activation='relu')(h1)
    
    lig_input = Input(shape=dim2)
    h2 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(lig_input)
    h2 = Conv2D(32, kernel_size=(5, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(h2)
    
    h3 = Concatenate(axis=1)([h1,h2])
    
    h3 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same', 
                activation='relu')(h3)
    h3 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(h3)
    h3 = Flatten()(h3)
    h3 = Dense(32, activation='relu')(h3)
    h3 = Dense(32, activation='relu')(h3)
    output = Dense(2, activation='softmax')(h3)
    model = Model(inputs=[pro_input, lig_input], outputs=output)
    return model


#%%

# Load single input data

num_pairs = 3000

pos_dir = '../data/16_8/positive_x/*'
neg_dir = '../data/16_8/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X = X[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X = np.concatenate((X,X_neg), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_neg), axis=0)

print('X.shape:', X.shape)

# Create Y: Dock-(1,0), No dock-(0,1)
Y = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y.shape:', Y.shape)


#%%

batch_size = 100
n_epoch = 20

# CNN
cnn = create_cnn(dim=(24,4,1))
print(cnn.summary())

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn_hist = cnn.fit(X,Y,
               epochs=n_epoch,
               validation_split=0.1,
               verbose=1,
               batch_size=batch_size)

# ## 学習結果の確認
print('\nCNN Validation Loss:', cnn_hist.history['val_loss'][-1])
print('\nCNN Validation Accuracy:', cnn_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(cnn_hist)
plot_history_acc(cnn_hist)

#%%

batch_size = 100
n_epoch = 20

mlp = create_mlp(dim=(24,4,1))
print(mlp.summary())

mlp.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
mlp_hist = mlp.fit(X,Y,
                   epochs=n_epoch,
                   validation_split=0.1,
                   verbose=1,
                   batch_size=batch_size)

# ## 学習結果の確認
print('\nMLP Validation Loss:', mlp_hist.history['val_loss'][-1])
print('\nMLP Validation Accuracy:', mlp_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(mlp_hist)
plot_history_acc(mlp_hist)

#%%

batch_size = 200
n_epoch = 50

mlp3 = create_mlp3(dim=(24,4,1))
print(mlp3.summary())

mlp3.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
mlp3_hist = mlp3.fit(X_train, Y_train,
                     epochs=n_epoch,
                     validation_split=0.2,
                     verbose=1,
                     batch_size=batch_size)

# ## 学習結果の確認
print('MLP3 Validation Loss:', mlp3_hist.history['val_loss'][-1])
print('MLP3 Validation Accuracy:', mlp3_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(mlp3_hist)
plot_history_acc(mlp3_hist)

#%%

# Load multiple input training data

num_pairs = 3000
split = 16

pos_dir = '../data/16_8/positive_x/*'
neg_dir = '../data/16_8/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X_pos = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_pos = X_pos[np.newaxis,:,:,np.newaxis]
X_pro = X_pos[:,:split,:,:]
X_lig = X_pos[:,split:,:,:]

# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X_neg_pro = X_neg[:,:split,:,:]
X_neg_lig = X_neg[:,split:,:,:]
X_pro = np.concatenate((X_pro,X_neg_pro), axis=0)
X_lig = np.concatenate((X_lig,X_neg_lig), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X_pos_pro = X_pos[:,:split,:,:]
    X_pos_lig = X_pos[:,split:,:,:]
    X_pro = np.concatenate((X_pro,X_pos_pro), axis=0)
    X_lig = np.concatenate((X_lig,X_pos_lig), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X_neg_pro = X_neg[:,:split,:,:]
    X_neg_lig = X_neg[:,split:,:,:]
    X_pro = np.concatenate((X_pro,X_neg_pro), axis=0)
    X_lig = np.concatenate((X_lig,X_neg_lig), axis=0)

print('X_pro.shape:', X_pro.shape)
print('X_lig.shape:', X_lig.shape)

# Create Y: Dock-(1,0), No dock-(0,1)
Y_train = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y_train.shape:', Y_train.shape)

#%%

batch_size = 20
n_epoch = 30

multiple_cnn = create_multiple_cnn(dim1=(16,4,1), dim2=(8,4,1))
print(multiple_cnn.summary())

multiple_cnn.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
multiple_cnn_hist = multiple_cnn.fit([X_pro,X_lig], Y_train,
                                     epochs=n_epoch,
                                     validation_split=0.2,
                                     verbose=1,
                                     batch_size=batch_size)

# ## 学習結果の確認
print('CNN: Mult Validation Loss:', multiple_cnn_hist.history['val_loss'][-1])
print('CNN: Mult Validation Accuracy:', multiple_cnn_hist.history['val_categorical_accuracy'][-1])

plot_history_loss(multiple_cnn_hist)
plot_history_acc(multiple_cnn_hist)

#%%

# Load LSTM training data

num_pairs = 3000

pos_dir = '../data/6_6/positive_x/*'
neg_dir = '../data/6_6/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X_train = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_train = X_train[np.newaxis,:,:]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:]
X_train = np.concatenate((X_train,X_neg), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:]
    X_train = np.concatenate((X_train,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:]
    X_train = np.concatenate((X_train,X_neg), axis=0)

print('X_train.shape:', X_train.shape)

# Create Y: Dock-(1,0), No dock-(0,1)
Y_train = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y_train.shape:', Y_train.shape)

#%%

mlp_mult = create_mlp_mult(dim1=(40,4,1), dim2=(8,4,1))
print(mlp_mult.summary())

mlp_mult.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
mlp_mult_hist = mlp_mult.fit([X_pro,X_lig], Y_train,
                             epochs=n_epoch,
                             validation_split=0.2,
                             verbose=1,
                             batch_size=batch_size)

# ## 学習結果の確認
print('MLP (Mult) Validation Loss:', mlp_mult_hist.history['val_loss'][-1])
print('MLP (Mult) Validation Accuracy:', mlp_mult_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(mlp_mult_hist)
plot_history_acc(mlp_mult_hist)

#%%

lstm = create_lstm(12)
print(lstm.summary())

lstm.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
lstm_hist = lstm.fit(X_train, Y_train,
                     epochs=n_epoch,
                     validation_split=0.3,
                     verbose=1,
                     batch_size=batch_size)

# ## 学習結果の確認
print('CNN: Mult Validation Loss:', lstm_hist.history['val_loss'][-1])
print('CNN: Mult Validation Accuracy:', lstm.history['val_categorical_accuracy'][-1])
plot_history_loss(lstm_hist)
plot_history_acc(lstm_hist)

#%%

# Extraction test

predicted_cnn = np.random.randn(20,2)
pro_id = np.repeat(range(1,3),10)
lig_id = np.tile(range(1,11),2)
pair_id = pd.DataFrame({'pro_id':pro_id, 'lig_id':lig_id}, columns=['pro_id','lig_id'])

prob = pd.DataFrame(predicted_cnn, columns=['dock','no_dock'])
predict = pd.concat([pair_id, prob], axis=1)
predict.to_csv('../fig/predict.csv', index=False)

test_predictions = np.arange(1,12).reshape(1,-1)
for i in range(1,num_pro+1):
    top_10 = predict[predict.pro_id==i].sort_values(by='dock', ascending=False)[:10]
    lig_row = np.concatenate((np.array([i]), top_10.lig_id.values))[np.newaxis,:]
    test_predictions = np.concatenate((test_predictions,lig_row), axis=0)
    
lig_predict = pd.DataFrame(test_predictions[1:],
                           columns=['pro_id','lig1_id','lig2_id','lig3_id','lig4_id','lig5_id',
                                     'lig6_id','lig7_id','lig8_id','lig9_id','lig10_id'])
    
lig_predict.to_csv('../fig/test_predictions.txt', sep='\t', index=False)