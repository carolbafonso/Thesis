#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:51:08 2020

@author: carol
"""

import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
import keras.backend as K

from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
from scipy.signal import butter, lfilter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):

    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=0)
    return y

################            
# AUTO-ENCODER #
################
            
def autoencoder(M, X_train):
    print("M=", M)
    x = Input(shape=(M,))
    print("x=", x)
    b = Dense(int(M/2), activation='tanh')(x)
    c = Dense(int(M/3), activation='tanh')(b)
    d = Dense(int(M/6), activation='tanh')(c)
    e = Dense(int(M/3), activation='tanh')(d)
    f = Dense(int(M/2), activation='tanh')(e)
    y = Dense(M, activation='tanh')(f)

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model = Model(x,y)
    optim = optimizers.Adam(lr=0.0005)
    model.compile(loss='mean_squared_error', optimizer=optim, metrics=['accuracy', mean_pred])
    
    predict = model.predict(X_train)

    Low_filt = butter_lowpass_filter(predict, 1, 1000, 5)
    High_filt = butter_highpass_filter(Low_filt, 40, 1000, 5)    
    
    fig, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(10,5))
    ax1.plot(np.linspace(0,len(High_filt),num=len(High_filt)), High_filt, color='lightblue', label='Normal Signal')
    
    model.summary()
    
    history = model.fit(X_train, X_train, batch_size=32, epochs=200, verbose=1, validation_split=0.2)
    
    plt.figure()
    x_aux = range(1, len(history.history['loss'])+1)
    train, = plt.plot(x_aux, history.history['loss'], 'b', label = 'Train')
    plt.ylim(ymin=0.0)
    plt.legend(handles=[train])
    plt.grid()
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure()
    x_aux = range(1, len(history.history['accuracy'])+1)
    train, = plt.plot(x_aux, history.history['accuracy'], 'green', label = 'Train')
    plt.ylim(ymin=0.0)
    plt.legend(handles=[train])
    plt.grid()
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.show() 
    
    return model, history


########            
# MAIN #
########
    
filename = 'data_1.pk'

with open(filename, 'rb') as hf:
    data = pickle.load(hf)
    

for i in data[0]:
    j = 0
    #while j < 15:
    M = len(data[0]['0'][0]) #data[{train}, {test}][lead][vetor da lead]
    Train = data[0]['0']
    plt.plot(data[0]['0'][0])
    plt.show()
    model, history = autoencoder(M, Train)
    sys.exit()
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
