import os
import sys
import numpy as np
import librosa
from scipy.io import wavfile

path = "C:\\Users\\Shivank\\Desktop\\RnnSpeech\\test"


sample_rate1, sample1 = wavfile.read("C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train\\bed_0.wav")
print('file loaded,'+(str)(sample_rate1))
print(sample1.shape)
mfccs1 = librosa.feature.mfcc(y=sample1, sr=sample_rate1, n_mfcc=5)
#print(mfccs1)
mfccs1 = np.transpose(mfccs1) # ( 32, 5 )
print(mfccs1)
mfccs1 = np.lib.pad(mfccs1,((0,32-mfccs1.shape[0]),(0,0)), mode = 'constant', constant_values=0)
print(mfccs1.shape)

sample_rate2, sample2 = wavfile.read("C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train\\bed_2.wav")
print('file loaded,'+(str)(sample_rate2))
print(sample2.shape)
mfccs2 = librosa.feature.mfcc(y=sample2, sr=sample_rate2, n_mfcc=5)
#print(mfccs2)
mfccs2 = np.transpose(mfccs2)
mfccs2 = np.lib.pad(mfccs2,((0,32-mfccs2.shape[0]),(0,0)), mode='constant', constant_values=0)
print(mfccs2)
print(mfccs2.shape)

print(np.stack((mfccs1,mfccs2)).shape)

'''
max=0
i = 1
files = [file for file in os.listdir(path)]
for file_name in files:
    unit = os.path.join(path,(str)(file_name))
    sample_rate, sample = wavfile.read(unit)

    print((str)(i)+" files scanned. ")
    i = i + 1

    mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=26)
    mfccs = np.transpose(mfccs)

    if max < mfccs.shape[0]:
        max = mfccs.shape[0]

print(max)
'''
