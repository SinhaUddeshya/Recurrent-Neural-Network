from python_speech_features import mfcc, logfbank
import numpy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from pylab import *

sample_rate, sample = wavfile.read("C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train\\bed_1.wav")
print('file loaded,'+(str)(sample_rate))
print(sample[0:100])
sampleDataType = sample.dtype
sample = sample/(2.**15)
print(sample.shape)
sampleShape = sample.shape
samplePoints = float(sample.shape[0])
signalDuration = sample.shape[0] / sample_rate
timeArray = numpy.arange(0, samplePoints, 1)
timeArray = timeArray / sample_rate
timeArray = timeArray * 1000

plt.plot(timeArray, sample, color='G')
plt.ylabel('Amplitude')
plt.xlabel('Time [msec]')
print('showing plot')
plt.show()

mfcc_feat = mfcc(sample ,sample_rate, numcep=13, nfilt=13)
fbank_feat = logfbank(sample,sample_rate, nfilt=13)

print(fbank_feat[1:3,:])
