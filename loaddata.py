import numpy as np
# from keras.utils import np_utils
from statsmodels.tools import categorical

nb_classes = 30

test_X = np.load('test_features.npy')       
test_labels = np.load('test_labels.npy')
# print(test_X.shape)
# print(test_labels.shape)

training_X = np.load('train_features.npy')
training_labels = np.load('train_labels.npy')
# print(training_X.shape)
# print(training_labels.shape)

test_Y = categorical(test_labels, drop=True)
training_Y = categorical(training_labels, drop=True)
# print(test_Y.shape)
# print(training_Y.shape)
