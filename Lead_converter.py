"""
Created on Wed Oct  4 15:40:10 2019

@author: carol
"""

import matplotlib.pyplot as plt
import scipy.io
import glob
import numpy as np
import pickle

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
        
        if l > np_array[0] or h < np_array[d-1]:
           sig = np_array[l:h]
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
mat_t = []
count = 0
for f in files:
    count += 1
    pf = scipy.io.loadmat(f)
    if count <= 349:
        #train
        mat.append(pf)
    else:
        #test
        mat_t.append(pf)

#filter
fs = 1000
low = 1
high = 40
filename = 'data_1.pk'

#DATA#
def data_sig(mat):
    
    dim = len(mat[0]['val'])
    
    X_data = {'0':[], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
               '9': [], '10': [], '11': [],'12': [],'13': [],'14': []}
    i=0
    count = 0
    while i < len(mat):
        j = 0
        count += 1
        print("new = ", count)
        while j < dim:
            sig = mat[i]['val'][j] #mat[sample]['val'][lead][window]
            
            #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
            
            Low_filt = butter_lowpass_filter(sig, high, fs, 5)
            High_filt = butter_highpass_filter(Low_filt, low, fs, 5) 
                
            #ax1.plot(np.linspace(0,len(sig),num=len(sig)), sig, color='lightblue', label='Normal Signal')
            #ax2.plot(np.linspace(0,len(sig),num=len(High_filt)), High_filt, color='black', label='Filtered signal')
            
            r_peaks = detectors.pan_tompkins_detector(sig)
            
            r_peaks = np.array(r_peaks)
            
            ax3 = cut_signal(High_filt,r_peaks, fs) 
            
            #print("tamanhos= ", len(mat[i]['val'][j]))
            
            if (len(ax3) == 650):
                X_data[str(j)].append(ax3)
            
            j += 1
        
        i += 1 
    
    for j in range(15):
        X_data[str(j)] = np.array(X_data[str(j)])

    return X_data

detectors = Detectors(fs)

X_train = data_sig(mat)
X_test = data_sig(mat_t)
    

with open(filename, 'wb') as hf:
    pickle.dump((X_train, X_test), hf)
   
with open(filename, 'rb') as hf:
    data = pickle.load(hf)
   


    
