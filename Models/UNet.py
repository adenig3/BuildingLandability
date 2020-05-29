import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers


""" Encoder/Decoder architecture using standard layers"""
""" Can use different dropout rates, as well as batch norms"""

class UNet:
    def __init__(self, num_filters, image_size, lamb, dropout_rates, batchnorm_bottleneck):
        self.num_filters = num_filters
        self.image_size = image_size
        self.lamb = lamb
        self.dropdown = dropout_rates[0]
        self.dropup = dropout_rates[1]
        self.batchnorm_before_bottleneck = batchnorm_bottleneck[0]
        self.batchnorm_ater_bottleneck = batchnorm_bottleneck[1]

    def configure(self):
        if len(self.num_filters) != 5:
            raise Exception("Improper Number of Filters. Alter Model Architecture")
        inputs = keras.layers.Input((self.image_size, self.image_size, 3))
        p0 = inputs
        c1, p1 = self.down_block(p0, self.num_filters[0])
        c2, p2 = self.down_block(p1, self.num_filters[1])
        c3, p3 = self.down_block(p2, self.num_filters[2])
        c4, p4 = self.down_block(p3, self.num_filters[3])

        bn = self.bottleneck(p4, self.num_filters[4])

        u1 = self.up_block(bn, c4, self.num_filters[3])
        u2 = self.up_block(u1, c3, self.num_filters[2])
        u3 = self.up_block(u2, c2, self.num_filters[1])
        u4 = self.up_block(u3, c1, self.num_filters[0])

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        model = keras.models.Model(inputs, outputs)
        return model

    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=regularizers.l2(self.lamb))(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=regularizers.l2(self.lamb))(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        p = keras.layers.Dropout(self.dropdown)(p)
        return c, p

    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=regularizers.l2(self.lamb))(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=regularizers.l2(self.lamb))(c)
        c = keras.layers.Dropout(self.dropup)(c)
        return c

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = x
        if self.batchnorm_before_bottleneck:
            c = keras.layers.BatchNormalization()(c)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if self.batchnorm_ater_bottleneck:
            c = keras.layers.BatchNormalization()(c)
        return c