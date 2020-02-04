"""
Created on Wed Oct  4 15:40:10 2019

@author: carol
"""

import matplotlib.pyplot as plt
import scipy.io
import glob
import numpy as np

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
    #cut window [p_max-0.2:p_max+0.4]
    
<<<<<<< HEAD
    sig = np.zeros((len(peaks), 650))
    
    for pp in range(len(peaks)):
        p_max = peaks[pp]
=======
    for pp in peaks:
        p_max = pp
>>>>>>> eda566e5be7b56dcb6c045496fcbbcbccb68faa3
        d = len(np_array)
        l = int(p_max - fs*0.2)
        h = int(p_max + fs*0.45)
        print("l1=", l)
        print("h1=" , h)
        print("array[0] = ", np_array)
        
<<<<<<< HEAD
        if l >= 0 and h < d:
            sig[pp] = np_array[l:h]
            #TODO: Normalizaçao [-1, 1]
            #print("sig=", sig)
            plt.plot(np.linspace(l, h, num=len(sig)), sig)
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
    optim = optimizers.Adam(lr=0.00001)
    model.compile(loss='mean_squared_error', optimizer=optim)
    
    model.summary()
    
    history = model.fit(X_train, X_train, batch_size=16, epochs=200, verbose=1, validation_split=0.2)
    
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
=======
        if l > np_array[0] or h < np_array[d-1]:
            sig = np_array[l:h]
            #print("sig=", sig)
            plt.plot(np.linspace(l, h, num=len(sig)), sig)
            plt.show()
            
>>>>>>> eda566e5be7b56dcb6c045496fcbbcbccb68faa3

########
# MAIN #     
########
    
#data
files = [file for file in glob.glob('PTB/PTB/*.mat')]
mat = []
for f in files:
    pf = scipy.io.loadmat(f)
    mat.append(pf)

#filter
fs = 1000
low = 1
high = 40
dim = len(mat[0]['val'])

X_train = []

detectors = Detectors(fs)

<<<<<<< HEAD
i=0
#FOR EVERY FILE:   <<<<<<<<<

# FIRST: DETECT PEAKS IN LEAD 1
# SECOND: CUT HEARTBEATS IN EVERY LEAD AT LOCATIONS DETECTED AT LEAD 1
while i < 1:
    sig = mat[0]['val'][i] #mat[sample]['val'][lead][window]
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(10,5))
=======
      

while i < dim:
    sig = mat[0]['val'][i][2000:4000] #mat[sample]['val'][lead][window]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
    
    Low_filt = butter_lowpass_filter(sig, high, fs, 5)
    High_filt = butter_highpass_filter(Low_filt, low, fs, 5) 
>>>>>>> eda566e5be7b56dcb6c045496fcbbcbccb68faa3
    
    Low_filt = butter_lowpass_filter(sig, high, fs, 5)
    High_filt = butter_highpass_filter(Low_filt, low, fs, 5) 
        
    #ax1.plot(np.linspace(0,len(sig),num=len(sig)), sig, color='lightblue', label='Normal Signal')
    #ax2.plot(np.linspace(0,len(sig),num=len(High_filt)), High_filt, color='black', label='Filtered signal')
    
    r_peaks = detectors.pan_tompkins_detector(sig)
    print("r_peaks=", r_peaks)
    
    r_peaks = np.array(r_peaks)
    
    ax3 = cut_signal(High_filt,r_peaks, fs) 
    
    sig_train = 2*(ax3 - np.min(ax3))/(np.max(ax3) - np.min(ax3)) - 1
    
    # X_train = np.concatenate((X_train, ax3))   <<<<
    
# NO fim de ir buscar picos a todos os ficheiros:
    X_train = sig_train

    print(X_train.shape)
    model, history = autoencoder(len(X_train[1]), X_train)
    
    print(history.history['loss'])
    
    i += 1
