import os
import sys
import numpy as np
import librosa
from scipy.io import wavfile
import pandas as pd

def parser(path):

    # function to load files and extract features
    files = [file for file in os.listdir(path)]
    np.random.shuffle(files)
    print('listed: '+(str)(path))

    count = 0
    i = 0
    features = np.array([], dtype='float32')
    label = ["" for x in range((round)(len(files)))]

    for file_name in files: # for all files, extract features and labels

        #if i>:
        #    break

        unit = os.path.join(path,(str)(file_name))
        sample_rate, sample = wavfile.read(unit)
        print("Loaded")

        name = file_name[:file_name.index('_')]
        label[i] = name

        stft = np.abs(librosa.stft(sample))
        #print("Loaded stft: ",stft.shape)
        mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=26).T, axis=0)
        #print("Loaded mfccs: ",mfccs.shape)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        #print("Loaded chroma: ",chroma.shape)
        mel = np.mean(librosa.feature.melspectrogram(sample, sr=sample_rate).T,axis=0)
        #print("Loaded mel: ",mel.shape)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        #print("Loaded contrast: ",contrast.shape)
        #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(sample),sr=sample_rate)
        #print("Loaded tonnetz: ",tonnetz.shape)

        print((str)(i+1)+" files done... ")
        i = i + 1
        ext_features = np.hstack([mfccs, chroma, mel, contrast])

        if count == 0:
            count = 1
            features = ext_features
        else:
            features = np.vstack((features, ext_features))

        labels = np.asarray(label) # ( number of files, )

        #np.save(((str)(os.path.basename(path)))+'_features', features)
        #print('features saved')
        #np.save(((str)(os.path.basename(path)))+'_labels', labels)
        #print('labels saved')

        #print(features.shape)

    np.save(((str)(os.path.basename(path)))+'_features', features)
    print('features saved')
    np.save(((str)(os.path.basename(path)))+'_labels', labels)
    print('labels saved')

parser("C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train")
