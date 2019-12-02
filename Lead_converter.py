import matplotlib.pyplot as plt
import scipy.io
import glob
import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def findPeaks(signal):
    find_peaks(signal, height=0)
    peaks, _ = find_peaks(signal)
    plt.plot(signal, color='grey')
    plt.plot(peaks, signal[peaks], "x")
   # plt.plot(peaks, signal[peaks], 'o', color='red')
    plt.show()
    
def cut_signal(mat,i):
    sig = mat[0]['val'][i][2000:2750]
    sig_f = butter_lowpass_filter(sig, 2.5, 100.0, 5) 
    findPeaks(sig_f)
    #plt.plot(sig_f_p, color='grey')
    #plt.show()
    
   
        

files = [file for file in glob.glob('PTB/PTB/*.mat')]
mat = []
for f in files:
    pf = scipy.io.loadmat(f)
    mat.append(pf)

#lowpass filter
fs = 5000.0
low = 1000.0
high = 1250.0
i = 0
dim = len(mat[0]['val'])

"""
sig = mat[0]['val'][1][2000:4000]
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))
ax = plt.axhline(y=0, color='r', linestyle='-')
filtered = butter_lowpass_filter(sig, 2.5, 100.0, 5)
ax1.plot(np.linspace(10,15,num=len(sig)), sig+500, color='lightblue', label='Normal Signal')
ax2 = findPeaks(filtered)
cut(mat)
"""


while i < dim:
    sig = mat[0]['val'][i][2000:4000]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))
    ax = plt.axhline(y=0, color='r', linestyle='-')
    filtered = butter_lowpass_filter(sig, 2.5, 100.0, 5)
    ax1.plot(np.linspace(10,15,num=len(sig)), sig+500, color='lightblue', label='Normal Signal')
    ax2 = findPeaks(filtered)
    cut(mat,i)
    i += 1
