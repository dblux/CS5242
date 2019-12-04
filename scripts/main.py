#!/usr/bin/env python3
#%%

import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers
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

def create_mlp3(dim):
    im_input = Input(shape=dim)
    h = Flatten()(im_input)
    h = Dense(100, activation='relu')(h)
    h = Dense(100, activation='relu')(h)
    output = Dense(2, activation='softmax')(h)
    model = Model(inputs=im_input, outputs=output)
    
    return model

#model.add(BatchNormalization())
#-----------
# CNNモデル
#-----------
def create_cnn(dim):
    model = Sequential()
    
    model.add(Conv2D(8, input_shape=dim,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(8,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model

def create_cnn2(dim):
    im_input = Input(shape=dim)
    # Due to memory limitation, images will resized on-the-fly.
    #x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(im_input)
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
    output = Dense(2, activation='softmax', name='predictions')(x)
    model = Model(inputs=im_input, outputs=output)
    
    return model

def plot_history_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist.history['loss'],label="Training set")
    plt.plot(hist.history['val_loss'],label="Validation set")
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_history_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist.history['categorical_accuracy'],label="Training set")
    plt.plot(hist.history['val_categorical_accuracy'],label="Validation set")
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.tight_layout()
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

# Load testing data

print('Loading X_test...\n')
test_dir = '../data/16_8/test_x/*'
test_files_path = sorted(glob.glob(test_dir))
#num_pairs = len(test_files_path)
num_pairs = 10

# Load first sample
X_test = np.loadtxt(test_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_test = X_test[np.newaxis,:,:,np.newaxis]

for i in range(1,num_pairs):
    # Concat positive sample
    X_1 = np.loadtxt(test_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_1 = X_1[np.newaxis,:,:,np.newaxis]
    X_test = np.concatenate((X_test,X_1), axis=0)

#%%

batch_size = 150
n_epoch = 150

# CNN
cnn = create_cnn2(dim=(24,4,1))
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

# ## 学習結果の確認
print('CNN Validation Loss:', cnn_hist.history['val_loss'][-1])
print('CNN Validation Accuracy:', cnn_hist.history['val_categorical_accuracy'][-1])

with open('../fig/cnn_history.pkl', 'wb') as file:
    pickle.dump(cnn_hist.history, file)
    
plot_history_loss(cnn_hist)
#plt.savefig('../fig/cnn_loss.png', dpi=150)
plot_history_acc(cnn_hist)
#plt.savefig('../fig/cnn_acc.png', dpi=150)

#%%

batch_size = 50
n_epoch = 30

# VGGNet

cnn2 = create_cnn2(dim=(24,4,1))
print(cnn2.summary())

sgd = optimizers.SGD(lr=0.01,
                     momentum=0.9,
                     decay=5e-4)

cnn2.compile(optimizer=sgd,
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn2_hist = cnn2.fit(X_train,Y_train,
                     epochs=n_epoch,
                     validation_split=0.2,
                     verbose=1,
                     batch_size=batch_size)

# ## 学習結果の確認
print('CNN2 Validation Loss:', cnn2_hist.history['val_loss'][-1])
print('CNN2 Validation Accuracy:', cnn2_hist.history['val_categorical_accuracy'][-1])

with open('../fig/cnn2_history.pkl', 'wb') as file:
    pickle.dump(cnn2_hist.history, file)
    
#with open('../fig/cnn_history.pkl', 'rb') as file:
#    cnn_train_data = pickle.load(file)

plot_history_loss(cnn2_hist)
#plt.savefig('../fig/cnn_loss.png', dpi=150)
plot_history_acc(cnn2_hist)
#plt.savefig('../fig/cnn_acc.png', dpi=150)

#%%

batch_size = 200
n_epoch = 150

mlp = create_mlp(dim=(16,4,1))
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

# ## 学習結果の確認
print('MLP Validation Loss:', mlp_hist.history['val_loss'][-1])
print('MLP Validation Accuracy:', mlp_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(mlp_hist)
plot_history_acc(mlp_hist)

#%%

# Create df to store results

pro_id = np.repeat(range(1,825),824)
lig_id = np.tile(range(1,825),824)
pair_id = pd.DataFrame({'pro_id':pro_id, 'lig_id':lig_id}, columns=['pro_id','lig_id'])

predicted_cnn = cnn.predict(X_test, batch_size)
print(predicted_cnn)
prob = pd.DataFrame(predicted_cnn, columns=['dock','no_dock'])
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