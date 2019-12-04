#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

def plot_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist['loss'],label="Training data set")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('../fig/final_loss.png', dpi=150)
    plt.show()

def plot_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.figure(figsize=(6.4,4.0))
    plt.plot(hist['categorical_accuracy'],label="Training data set")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.tight_layout()
    plt.savefig('../fig/final_acc.png', dpi=150)
    plt.show()
    
#%%
    
# Create df to store results

lig_id = np.load('../fig/lig_predictions.npy')[1:]
pro_id = np.arange(1,825)[:,np.newaxis]
pred_arr = np.concatenate((pro_id,lig_id), axis=1)
lig_predict = pd.DataFrame(pred_arr,
                           columns=['pro_id','lig1_id','lig2_id','lig3_id','lig4_id','lig5_id',
                                    'lig6_id','lig7_id','lig8_id','lig9_id','lig10_id'])
    
lig_predict.to_csv('../fig/test_predictions.txt', sep='\t', index=False)

#%%

with open('../fig/cnn_history.pkl', 'rb') as file:
    cnn_fit = pickle.load(file)

cnn_fit['categorical_accuracy']

plot_acc(cnn_fit)
plot_loss(cnn_fit)