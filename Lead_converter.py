"""
Created on Wed Oct  4 15:40:10 2019

@author: carol
"""

import matplotlib.pyplot as plt
import scipy.io
import glob
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
from detectors_mod import Detectors
from trahanias import trahanias

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
            plt.plot(np.linspace(l, h, num=len(sig)), sig)
            plt.show()
            

#########   
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
i = 0
dim = len(mat[0]['val'])

new_matrix = []

detectors = Detectors(fs)

      

while i < dim:
    sig = mat[0]['val'][i][2000:4000] #mat[sample]['val'][lead][window]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
    
    Low_filt = butter_lowpass_filter(sig, high, fs, 5)
    High_filt = butter_highpass_filter(Low_filt, low, fs, 5) 
    
    ax1.plot(np.linspace(0,len(sig),num=len(sig)), sig, color='lightblue', label='Normal Signal')
    ax2.plot(np.linspace(0,len(sig),num=len(High_filt)), High_filt, color='black', label='Filtered signal')
    
    r_peaks = detectors.pan_tompkins_detector(sig)
    #r_peaks = trahanias(High_filt, fs)
    print("r_peaks=", r_peaks)
    
    r_peaks = np.array(r_peaks)
    
    ax3 = cut_signal(High_filt,r_peaks, fs) 
    
    i += 1
