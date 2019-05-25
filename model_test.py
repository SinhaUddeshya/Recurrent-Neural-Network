import numpy as np

import keras
import keras.backend as K

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, TimeDistributed, Activation, SimpleRNN
#from keras.layers.recurrent import LSTM

from loaddata import test_X, training_X, test_Y, training_Y

nb_classes = 30
batch_size = 512
num_hiddens = 100

'''
training_X = training_X.reshape(training_X.shape[0], -1, 1)
test_X = test_X.reshape(test_X.shape[0], -1, 1)
training_X = training_X.astype('float32')
test_X = test_X.astype('float32')
training_X /= 255
test_X /=255
'''

print('training_X: ',training_X.shape) # (46114,32,26)
print('training_Y: ',training_Y.shape) # (46114,30)
print('test_X: ',test_X.shape)         # (2427,32,26)
print('test_Y: ',test_Y.shape)         # (2427,30)

model = Sequential()

model.add(Dense(units=num_hiddens, input_shape=training_X.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(training_X, training_Y, batch_size=batch_size, epochs=50, validation_data=(test_X,test_Y), verbose=1)
score, accuracy = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)

print('Test score: ', score)
print('Test accuracy:', accuracy)
