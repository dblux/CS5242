#!/usr/bin/env python3
#%%

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model        
from keras.layers import Input, Dense, Conv2D, Activation, Dropout, Flatten, Concatenate, BatchNormalization, LeakyReLU

#%%

def create_mlp(dim):
    im_input = Input(shape=dim)
    h = Flatten()(im_input)
    h = Dense(200)(h)
    h = Dropout(0.4)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(200)(h)
    h = Dropout(0.1)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(96)(h)
    h = Dropout(0.1)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(96)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(48)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(48)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(24)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(24)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(16)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(16)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(8)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(8)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(8)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)
    h = Dense(2)(h)
    output = Activation('softmax')(h)
    model = Model(inputs=im_input, outputs=output)
    
    return model

def create_mlp1(dim):
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

def create_cnn(dim=(24,4,1)):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(8,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
   
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model

def cnn_yuki(dim=(24,4,1)):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(16,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(16,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(8,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(8,
                     kernel_size=(7, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
   
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
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model


def create_multiple_cnn(dim1=(16,4,1), dim2=(8,4,1)):
    pro_input = Input(shape=dim1)
    h1 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(pro_input)
    h1 = BatchNormalization()(h1)
    h1 = Dropout(0.3)(h1)
    
    h1 = Conv2D(32, kernel_size=(5, 3),
                strides=(3, 1),
                padding='valid',
                activation='relu')(h1)
    h1 = BatchNormalization()(h1)
    h1 = Dropout(0.3)(h1)
    
    lig_input = Input(shape=dim2)
    h2 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(lig_input)
    h2 = BatchNormalization()(h2)
    h2 = Dropout(0.3)(h2)
    h2 = Conv2D(32, kernel_size=(5, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(h2)
    h2 = BatchNormalization()(h2)
    h2 = Dropout(0.3)(h2)
    
    h3 = Concatenate(axis=2)([h1,h2])
    
    h3 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same', 
                activation='relu')(h3)
    h3 = BatchNormalization()(h3)
    h3 = Dropout(0.3)(h3)
    h3 = Conv2D(32, kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu')(h3)
    h3 = BatchNormalization()(h3)
    h3 = Dropout(0.3)(h3)
    h3 = Flatten()(h3)
    h3 = Dense(64, activation='relu')(h3)
    h3 = BatchNormalization()(h3)
#    h3 = Dropout(0.3)(h3)
    h3 = Dense(64, activation='relu')(h3)
    h3 = BatchNormalization()(h3)
#    h3 = Dropout(0.3)(h3)
    output = Dense(2, activation='softmax')(h3)
    model = Model(inputs=[pro_input, lig_input], outputs=output)
    return model

def plot_history_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist['loss'],label="Training data set")
    plt.plot(hist['val_loss'],label="Validation data set")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../fig/cnn_loss.png', dpi=150)
    plt.show()

def plot_history_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist['categorical_accuracy'],label="Training data set")
    plt.plot(hist['val_categorical_accuracy'],label="Validation data set")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../fig/cnn_acc.png', dpi=150)
    plt.show()

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

# Load multiple input training data

num_pairs = 3000
split = 16

pos_dir = '../data/16_8_3/positive_x/*'
neg_dir = '../data/16_8_3/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X_pos = np.load(pos_files_path[0])
X_pos = X_pos[np.newaxis,:,:,np.newaxis]
X_pro = X_pos[:,:split,:,:]
X_lig = X_pos[:,split:,:,:]

# Concat first negative sample
X_neg = np.load(neg_files_path[0])
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X_neg_pro = X_neg[:,:split,:,:]
X_neg_lig = X_neg[:,split:,:,:]
X_pro = np.concatenate((X_pro,X_neg_pro), axis=0)
X_lig = np.concatenate((X_lig,X_neg_lig), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.load(pos_files_path[i])
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X_pos_pro = X_pos[:,:split,:,:]
    X_pos_lig = X_pos[:,split:,:,:]
    X_pro = np.concatenate((X_pro,X_pos_pro), axis=0)
    X_lig = np.concatenate((X_lig,X_pos_lig), axis=0)
    # Concat negative sample
    X_neg = np.load(neg_files_path[i])
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

batch_size = 400
n_epoch = 200

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
                   batch_size=batch_size,
                   validation_split=0.1)

with open('../fig/mlp_history.pkl', 'wb') as file:
    pickle.dump(mlp_hist.history, file)

# ## 学習結果の確認
print('MLP Validation Loss:', mlp_hist.history['val_loss'][-1])
print('MLP Validation Accuracy:', mlp_hist.history['val_categorical_accuracy'][-1])

plot_history_loss(mlp_hist.history)
plot_history_acc(mlp_hist.history)

#%%

batch_size = 300
n_epoch = 400

cnn = create_cnn(dim=(24,4,1))
print(cnn.summary())

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn_hist = cnn.fit(X_train, Y_train,
                   epochs=n_epoch,
                   verbose=1,
                   batch_size=batch_size,
                   validation_split=0.1)

with open('../fig/cnn_history.pkl', 'wb') as file:
    pickle.dump(cnn_hist.history, file)

# ## 学習結果の確認
print('cnn1 Validation Loss:', cnn_hist.history['val_loss'][-1])
print('cnn1 Validation Accuracy:', cnn_hist.history['val_categorical_accuracy'][-1])

plot_history_loss(cnn_hist.history)

plot_history_acc(cnn_hist.history)


#%%

batch_size = 600
n_epoch = 200

multiple_cnn = create_multiple_cnn(dim1=(16,4,1), dim2=(8,4,1))
print(multiple_cnn.summary())

multiple_cnn.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
multiple_cnn_hist = multiple_cnn.fit([X_pro,X_lig], Y_train,
                                     epochs=n_epoch,
                                     validation_split=0.1,
                                     verbose=1,
                                     batch_size=batch_size)

# ## 学習結果の確認
print('CNN: Mult Validation Loss:', multiple_cnn_hist.history['val_loss'][-1])
print('CNN: Mult Validation Accuracy:', multiple_cnn_hist.history['val_categorical_accuracy'][-1])

plot_history_loss(multiple_cnn_hist)
plot_history_acc(multiple_cnn_hist)

#%%

# Load testing data

print('Loading X_test...\n')
test_dir = '../data/16_8_binary/test_x/'
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