
# coding: utf-8

# ## CNN

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model        
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten,BatchNormalization

#-----------
# CNNmodel
#-----------

def create_cnn(dim=(24,4,1)):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(8,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model
    

num_pairs = 3000

pos_dir = '../16_8/positive_x/*'
neg_dir = '../16_8/negative_x/*'

pro_name =list()
lig_name = list()

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X = X[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X = np.concatenate((X,X_neg), axis=0)

for i in range(0,num_pairs-1):
    text = str(pos_files_path[i])
    number =text[19:28]
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])
    text = str(neg_files_path[i])
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])

    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_neg), axis=0)

print('X.shape:', X.shape)


# In[2]:


pro_name =list()
lig_name = list()
for i in range(0,num_pairs):
    text = str(pos_files_path[i])
    number =text[19:28]
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])
    text2 = str(neg_files_path[i])
    number2 =text2[19:28]
    pro_name.append(number2.split("_")[0])
    lig_name.append(number2.split("_")[1])


# In[3]:


len(pro_name)


# In[4]:


# Create Y: Dock-(1,0), No dock-(0,1)
Y = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y.shape:', Y.shape)


# In[5]:


from sklearn.model_selection import train_test_split

indices = np.array(range(X.shape[0]))
X_train, X_test, Y_train, Y_test,pro_name_train,pro_name_test,lig_name_train,lig_name_test = train_test_split(X, Y, pro_name,lig_name,test_size=0.20, random_state=111)


# In[6]:


print(X_train.shape)
print(X_test.shape)


# In[7]:


# ミニバッチに含まれるサンプル数を指定
batch_size = 50
# epoch数を指定
n_epoch = 30

cnn = create_cnn(dim=(24,4,1))
print(cnn.summary())

cnn.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn_hist = cnn.fit(X_train,Y_train,
                   epochs=n_epoch,
                   validation_split=0.2,
                   verbose=1,
                   batch_size=batch_size)

print('\n', cnn_hist.history)


# In[8]:


## 学習結果の確認

def plot_history_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.figure()
    plt.plot(hist.history['loss'],label="Training ")
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def plot_history_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.figure()
    plt.plot(hist.history['categorical_accuracy'],label="Training set")
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.show()

plot_history_loss(cnn_hist)
plot_history_acc(cnn_hist)


# In[9]:


# テストデータを使ってモデルの評価
loss_and_accuracy = cnn.evaluate(X_train, Y_train, batch_size=128)
print(loss_and_accuracy)


# In[10]:


# テストデータを使ってモデルの評価
loss_and_accuracy = cnn.evaluate(X_test, Y_test, batch_size=128)
print(loss_and_accuracy)


# ## VGGnet

# In[12]:


import keras
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,TensorBoard

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder
from keras.datasets import cifar10
import cv2
# import gc
import numpy as np


# In[20]:


inputs = Input(shape=(24, 4, 1))
# Due to memory limitation, images will resized on-the-fly.
#x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
flattened = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(flattened)
x = Dropout(0.5, name='dropout1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='dropout2')(x)
# Change to 2 from 10
predictions = Dense(2, activation='softmax', name='predictions')(x)


BATCH_SIZE = 256
sgd = optimizers.SGD(lr=0.01,
                     momentum=0.9,
                     decay=5e-4)#, nesterov=False)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[21]:


# ミニバッチに含まれるサンプル数を指定
batch_size = 50
# epoch数を指定
n_epoch = 5

# 学習を開始します。
print('Training model...')
hist = model.fit(X_train,Y_train,
                   epochs=n_epoch,
                   validation_split=0.2,
                   verbose=1,
                   batch_size=batch_size)

print('\n', hist.history)


# ## CNN with dropout_BatchNormalization

# In[16]:


def create_cnn_drop(dim=(24,4,1)):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Conv2D(8,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model


# In[17]:


# ミニバッチに含まれるサンプル数を指定
batch_size = 50
# epoch数を指定
n_epoch = 30

cnn = create_cnn_drop(dim=(24,4,1))
print(cnn.summary())

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn_hist = cnn.fit(X_train,Y_train,
                   epochs=n_epoch,
                   validation_split=0.2,
                   verbose=1,
                   batch_size=batch_size)

print('\n', cnn_hist.history)


# In[18]:


# テストデータを使ってモデルの評価
loss_and_accuracy = cnn.evaluate(X_test, Y_test, batch_size=128)
print(loss_and_accuracy)


# ## For test data

# In[10]:


import pandas as pd
df = pd.DataFrame(np.random.random([len(Y_test), 4]), columns=['pro_id','lig_id','dock','no_dock'])

for i in range(len(Y_test)):
    df.pro_id[i]=pro_name[i]
    df.lig_id[i] = lig_name[i]
    pred_1 = cnn.predict(X_test[i].reshape(1, 308, 4, 1))[0]
    pred_2 = np.array(pred_1, dtype=str)
    df.dock[i] = pred_2[0]
    df.no_dock[i] = pred_2[1]


# In[ ]:


df


# In[ ]:


df.sort_values(by='dock', ascending=False)[:10]


# In[ ]:


df[df.pro_id==314.0].sort_values(by='dock', ascending=False)[:10]

