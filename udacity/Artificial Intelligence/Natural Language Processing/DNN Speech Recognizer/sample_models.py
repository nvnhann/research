from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech """
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model



def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech """
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu',name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech """
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn = GRU(units, return_sequences=True)(input_data)
    rnn = BatchNormalization()(rnn)
    for _ in range(recur_layers - 1):
        rnn = GRU(units, return_sequences=True)(rnn)
        rnn = BatchNormalization()(rnn)
    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech """
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True), merge_mode='concat')(input_data)
    bn_bidir_rnn = BatchNormalization()(bidir_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, cnn_filters, cnn_kernel_size, cnn_strides, output_dim=29):
    """ Build a deep network for speech """
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(cnn_filters, cnn_kernel_size, strides=cnn_strides, padding='same', activation='relu')(input_data)
    bn_cnn = BatchNormalization()(conv_1d)
    rnn = Bidirectional(GRU(units, return_sequences=True), merge_mode='concat')(bn_cnn)
    rnn = BatchNormalization()(rnn)
    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, cnn_kernel_size, 'same', cnn_strides)
    print(model.summary())
    return model