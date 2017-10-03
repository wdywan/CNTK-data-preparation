from scipy.io.wavfile import read as rd
from scipy.signal import butter, lfilter, zpk2sos, sosfilt
import scipy.signal as signal
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import math
import array
import struct
import wave
from math import sqrt
 
freq = 48000

def open_file(file):
    files = [f for f in listdir('%s' % file) if isfile(join('%s' % file, f))]
    return files
 
def write_file(file):
    files  = open('' + file + '.txt', 'w+')
    return files

def directories(dir, i):
    return dir[i]
   
def to_array(name):  
    a = np.asarray(((rd(name))[1])[:, 0], dtype = np.float32)   
    return a

def to_windows(name):
    final = []
    for n in name: # for every band in file
        window = []
        for i in range(0, 10000, 200):
            chunks = [n[x : x + freq] for x in range(i, 7*freq + i, freq)] # makes 350 windows 
            window += chunks
        final.append(window) 
    return final # return 10 bands made of 350 windows each one
 
def slice_array(name):
    sample = name
    chunks = np.array_split(sample, 10)
    return chunks
 
def calculate_average(name):
    average = np.mean(name)
    return average

def standard_deviation(name):
    deviation = np.std(name)
    return deviation
 
def root_mean_square(name):
    return np.sqrt(np.sum(np.square(name)) / len(name))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    z, p, k = butter(order, [low, high], btype='band', output='zpk')
    return z, p, k
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    z, p, k = butter_bandpass(lowcut, highcut, fs, order=order)
    convert = zpk2sos(z, p, k)
    y = sosfilt(convert, data)
    return y

def proccess(name):
    processed = []
    for j in range(len(lowcut)):    # applies 10 freq bands on copies of one window
        y = butter_bandpass_filter(name, lowcut[j], highcut[j], fs, order = 6)
        processed.append(y)

    return processed

def values_string(array):
    valuestr = ""
    for a in array:
        valuestr += (" " + str(a))
    return valuestr

lowcut = [150.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0]
highcut = [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0]
fs = 20000.0

labels = [100, 87, 34, 65, 21, 53, 18, 30, 56, 99, 11]
 
dirs = ['dir1', 'dir2', 'dir3']

for ln in range(len(dirs)):
    dir = directories(dirs, ln)
    rfiles = open_file(dir)
    np.set_printoptions(suppress = True)
    k = 0
    count = 0

    with open("train_data.txt", "a") as wfiles:
        with open("test_data.txt", "a") as wfiles2:
             for f in rfiles:            # for every file in current directory
                print (f)                # prints out name of a current file that is being processed
                inp_array = to_array(dir+ '/' + f) # saves .wav file as array

                inp_proccessed = proccess(inp_array)    # applies 10 band on copies of one array
                windows = to_windows(inp_proccessed)    # creates table of 10 bands made of 350 windows
                
                for i in range (0, 350):
                    values = [] 
                    take_windows = [windows[x][i] for x in range(0,10,1)] # takes one window from every band
                    for t in take_windows:
                        inp_slice = slice_array(t) # slices every band to 10 chunks 
                        rms = []
                        for j in inp_slice: 
                            rms.append(root_mean_square(j))         # calculates rms from every band
                        values.append(calculate_average(rms)/10)    # calulates avarage of rms
                        values.append(standard_deviation(rms)/10)   # calculates standard_deviation                        
                
                    valuestr = values_string(values)

                    if count % 10 != 0:
                        fname = wfiles
                    else:
                        fname = wfiles2

                    fname.write("|labels " + str(labels[k] / 10) + " |features" +  valuestr + "\n")
                    count += 1
                k += 1
wfiles.close()
wfiles2.close()
             