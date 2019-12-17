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

# detector pan-tompkins #
    
def detector_QRS(signal, cutoff, fs, order):
    filt = butter_lowpass_filter(signal, cutoff, fs, order)
    
    

# PEAKS #
def cut_signal(signal, peaks):
    #recive signal with peak r detected
    #cut window [p_max-0.2:p_max+0.4]
    
    p_max = max(peaks)
    l = p_max - 0.2
    h = p_max + 0.4
    dim_s = h-l
    plt.axis([l, h, 0, len(signal)])
    plt.axvline(x=l+0.1, color='red', linestyle='-')
    plt.axvline(x=h-0.1, color='blue', linestyle='-')
    plt.axhline(y=0, color='orange', linestyle='-')
    plt.plot(np.linspace(l,h,num=len(signal)), signal, color='black')
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

detectors = Detectors(fs)

while i < dim:
    sig = mat[0]['val'][i][2000:4000] #mat[sample]['val'][lead][window]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
    Low_filt = butter_lowpass_filter(sig, high, fs, 5)
    High_filt = butter_highpass_filter(Low_filt, high, fs, 5)
    ax1.plot(np.linspace(10,15,num=len(sig)), sig, color='lightblue', label='Normal Signal')
    ax2.plot(np.linspace(10,15,num=len(Low_filt)), Low_filt, color='black')
    r_peaks = detectors.pan_tompkins_detector(sig)
    print("r_peaks="), print(r_peaks)
    ax3 = cut_signal(High_filt, r_peaks)
    i += 1
