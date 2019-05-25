import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

sample_rate, sample = wavfile.read("C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train\\yes_1.wav")
print('file loaded,'+(str)(sample_rate))
print(sample.shape)
mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=26).T, axis=0)
print(mfccs.shape)
print(mfccs)
print(mfccs.strides)

'''plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs,x_axis='time')
plt.colorbar()
plt.title('mfccs')
plt.tight_layout()
plt.show()
'''


'''nfft = 320 # length of windowing segments
# fs = sample_rate   # Sampling frequency
pxx, freqs, bins, im = plt.specgram(sample, nfft, sample_rate)
plt.axis('on')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Window segments [20ms/320 samples per window ]')
plt.show()

sample = sample/(2.**15)
sampleShape = sample.shape
print("sample shape: "+(str)(sampleShape))
samplePoints = float(sample.shape[0])
print("sample points: "+(str)(samplePoints))
signalDuration = sample.shape[0] / sample_rate
print("signalDuration: "+(str)(signalDuration))
timeArray = numpy.arange(0, samplePoints, 1)
print('timeArray: '+(str)(timeArray))
timeArray = timeArray / sample_rate
timeArray = timeArray * 1000

plt.plot(timeArray, sample, color='G')
plt.ylabel('Amplitude')
plt.xlabel('Time [msec]')
print('showing plot')
plt.show()
'''
