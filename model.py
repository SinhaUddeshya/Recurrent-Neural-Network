import core.ctc_utils as ctc_utils
from utils.hparams import HParams

import keras
import keras.backend as K
from keras.initializations import uniform
from keras.activations import relu

from keras.models import Model

from keras.layers import Input
from keras.layers import GaussianNoise
from keras.layers import TimeDistributed
from keras.layers import Dense
from .layers import LSTM
from keras.layers import Masking
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import merge

from keras.regularizers import l1, l2, l1l2

from .layers import recurrent


def ctc_model(inputs, output, **kwargs):
    """ Given the input and output returns a model appending ctc_loss, the
    decoder, labels, and inputs_length
    # Arguments
        see core.ctc_utils.layer_utils.decode for more arguments
    """

    # Define placeholders
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    inputs_length = Input(name='inputs_length', shape=(None,), dtype='int32')

    # Define a decoder
    dec = Lambda(ctc_utils.decode, output_shape=ctc_utils.decode_output_shape,
                 arguments={'is_greedy': False}, name='decoder')
    y_pred = dec([output, inputs_length])

    ctc = Lambda(ctc_utils.ctc_lambda_func, output_shape=(1,), name="ctc")
    # Define loss as a layer
    loss = ctc([output, labels, inputs_length])

    return Model(input=[inputs, labels, inputs_length], output=[loss, y_pred])


def graves2006(num_features=26, num_hiddens=10, num_classes=28, std=.6):
    """ Implementation of Graves' model
    Reference:
        [1] Graves, Alex, et al. "Connectionist temporal classification:
        labelling unsegmented sequence data with recurrent neural networks."
        Proceedings of the 23rd international conference on Machine learning.
        ACM, 2006.
    """

    x = Input(name='inputs', shape=(None, num_features))
    o = x

    o = GaussianNoise(std)(o)
    o = Bidirectional(LSTM(num_hiddens,
                      return_sequences=True,
                      consume_less='gpu'))(o)
    o = TimeDistributed(Dense(num_classes))(o)

return ctc_model(x, o)
