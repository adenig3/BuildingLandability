import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers import *

""" Uses a VGG architecture for encoding and decoding
    Currently without pre-trained weights"""

class SegNet():
    def __init__(self, num_filters, image_size):
        self.image_size = image_size
        self.num_filters = num_filters


    def configure(self):
        inputs = keras.layers.Input((self.image_size, self.image_size, 3))
        e = self.encoder(inputs)
        outputs = self.decoder(e)
        model = keras.models.Model(inputs, outputs)
        return model


    def encoder(self, x): #IDK if i'm repeating the number of layers correctly
        x = Conv2D(self.num_filters[0], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[0], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(self.num_filters[1], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[1], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(self.num_filters[2], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[2], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[2], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(self.num_filters[3], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x


    def decoder(self, x):
        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[2], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(self.num_filters[2], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[2], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[1], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(self.num_filters[1], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_filters[0], kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(self.num_filters[0], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(self.num_filters[0], kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)

        #Match the output shape...IDK if this is right
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
        return outputs




