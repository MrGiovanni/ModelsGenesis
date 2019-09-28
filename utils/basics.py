from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
import tensorflow as tf

def conv_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                                 kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = Activation("relu")(x)
        return x
    return f

def conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f

def bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('bn_relu_conv'):
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
        return x
    return f

def atrous_conv_bn(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def atrous_conv_bn_relu(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f
