"""
Created on Wed Oct  4 15:40:10 2019

@author: carol
"""

import matplotlib.pyplot as plt
import scipy.io
import glob
import numpy as np
import sys

from scipy.signal import butter, lfilter, filtfilt
from detectors_mod import Detectors
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers


##############
# FUNCTIONS #
#############

# FILTERS #

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


# PEAKS #
def cut_signal(np_array, peaks, fs):
    #recive signal with peak r detected

    sig = np.zeros((len(peaks), 650))
    
    for pp in range(len(peaks)):
        p_max = peaks[pp]

    for pp in peaks:
        p_max = pp

        d = len(np_array)
        l = int(p_max - fs*0.2)
        h = int(p_max + fs*0.45)
        print("l1=", l)
        print("h1=" , h)
        print("array[0] = ", np_array)
        
        if l > np_array[0] or h < np_array[d-1]:
           sig = np_array[l:h]
           #print("sig=", sig)
           b = 2*(sig - np.min(sig))/(np.max(sig) - np.min(sig)) - 1
           plt.plot(np.linspace(l, h, num=len(b)), b)
           plt.show()
             
    return sig

    
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
    
    model = Model(x,y)
    optim = optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optim)
    
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
    
    return model, history


########
# MAIN #     
########
    
#data
files = [file for file in glob.glob('PTB/PTB/*.mat')]
mat = []

#filter
fs = 1000
low = 1
high = 40


X_train = list()
X_test = list()

detectors = Detectors(fs)

#FOR EVERY FILE:   <<<<<<<<<

# FIRST: DETECT PEAKS IN LEAD 1
# SECOND: CUT HEARTBEATS IN EVERY LEAD AT LOCATIONS DETECTED AT LEAD 1
for f in files:
    pf = scipy.io.loadmat(f)
    mat.append(pf)
    dim = len(mat[0]['val'])
    i = 0
    while i < dim:
        sig = mat[0]['val'][i] #mat[sample]['val'][lead][window]
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
        
        Low_filt = butter_lowpass_filter(sig, high, fs, 5)
        High_filt = butter_highpass_filter(Low_filt, low, fs, 5) 
            
        #ax1.plot(np.linspace(0,len(sig),num=len(sig)), sig, color='lightblue', label='Normal Signal')
        #ax2.plot(np.linspace(0,len(sig),num=len(High_filt)), High_filt, color='black', label='Filtered signal')
        
        r_peaks = detectors.pan_tompkins_detector(sig)
        print("r_peaks_lead1=", r_peaks)
        r_peaks = np.array(r_peaks)
        ax3 = cut_signal(High_filt,r_peaks, fs) 
        
        if(len(X_train) == 200):
            X_test.append(pf)
        else:
            X_train.append(ax3)  
        
        i += 1        

print("X_train", len(X_train))
print("X_test", len(X_test))
    
# NO fim de ir buscar picos a todos os ficheiros:
 
sys.exit()

X_train = np.array(X_train)
print(X_train.shape)

model, history = autoencoder((len(X_train[0])), X_train)

    
print(history.history['loss'])
    

