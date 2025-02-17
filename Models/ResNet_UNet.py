import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.initializers import *
""" Uses a ResNet 18 architecture for encoding and decoding
    Currently without pre-trained weights"""

class ResNet_18_UNet():
    def __init__(self, num_filters, image_size, lamb, dropout_rate):
        self.num_filters = num_filters
        self.image_size = image_size
        self.lamb = lamb
        self.dropout_rate = dropout_rate

    def encoder_identity_block(self, X, f, filters, stage, block):
            """
            Implementation of the identity block as defined in Figure 3

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network

            Returns:
            X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
            """

            # defining name basis
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            # Retrieve Filters
            F1, F2, F3 = filters

            # Save the input value. You'll need this later to add back to the main path.
            X_shortcut = X

            # First component of main path
            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
            X = Activation('relu')(X)


            # Second component of main path (≈3 lines)
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
            X = Activation('relu')(X)

            # Third component of main path (≈2 lines)
            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)


            return X

    #Works in reverse order?
    def decoder_identity_block(self, X, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters
