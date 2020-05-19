import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *

""" Uses a VGG architecture for encoding and decoding
    Currently without pre-trained weights"""

class VGG_UNet():
    def __init__(self, num_filters, image_size, lamb, dropout_rate):
        self.image_size = image_size
        self.num_filters = num_filters
        self.lamb = lamb
        self.dropout_rate = dropout_rate
    def configure(self):
        inputs = keras.layers.Input((self.image_size, self.image_size, 3))
        e, pool = self.encoder(inputs)
        bn = self.bottleneck(e[4])
        d = self.decoder(e, bn, pool)
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d)
        model = keras.models.Model(inputs, outputs)
        return model

    def conv_batch_relu(self, x, filter_num):
        x = Conv2D(filter_num, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(self.lamb))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        return x

    def encoder(self, x):
        x = self.conv_batch_relu(x, self.num_filters[0])
        x_1 = self.conv_batch_relu(x, self.num_filters[0])
        pool_1 = MaxPool2D(pool_size=(2,2))(x)

        x = self.conv_batch_relu(pool_1, self.num_filters[1])
        x_2 = self.conv_batch_relu(x, self.num_filters[1])
        pool_2 = MaxPool2D(pool_size=(2,2))(x)

        x = self.conv_batch_relu(pool_2, self.num_filters[2])
        x = self.conv_batch_relu(x, self.num_filters[2])
        x_3 = self.conv_batch_relu(x, self.num_filters[2])
        pool_3 = MaxPool2D(pool_size=(2, 2))(x)

        x = self.conv_batch_relu(pool_3, self.num_filters[3])
        x = self.conv_batch_relu(x, self.num_filters[3])
        x_4 = self.conv_batch_relu(x, self.num_filters[3])
        pool_4 = MaxPool2D(pool_size=(2, 2))(x)

        x = self.conv_batch_relu(pool_4, self.num_filters[4])
        x = self.conv_batch_relu(x, self.num_filters[4])
        x_5 = self.conv_batch_relu(x, self.num_filters[4])
        pool_5 = MaxPool2D(pool_size=(2, 2))(x) #This start the bottleneck

        x = [x_1, x_2, x_3, x_4, x_5]
        pool = [pool_1, pool_2, pool_3, pool_4, pool_5]
        return x, pool

    def bottleneck(self, x):
        x = self.conv_batch_relu(x,self.num_filters[4])
        return x

    def decoder(self, x, x_bottle, pool):
        x_1, x_2, x_3, x_4, x_5 = x
        pool_1, pool_2, pool_3, pool_4, pool_5 = pool
        x = UpSampling2D((2,2))(pool_5)
        x = Concatenate()([x, x_5]) #Combine with pre-pool
        x = self.conv_batch_relu(x, self.num_filters[4])
        x = self.conv_batch_relu(x, self.num_filters[4])
        x = self.conv_batch_relu(x, self.num_filters[4])

        x = UpSampling2D((2,2))(x)
        x = Concatenate()([x, x_4])
        x = self.conv_batch_relu(x, self.num_filters[3])
        x = self.conv_batch_relu(x, self.num_filters[3])
        x = self.conv_batch_relu(x, self.num_filters[3])

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_3])
        x = self.conv_batch_relu(x, self.num_filters[2])
        x = self.conv_batch_relu(x, self.num_filters[2])
        x = self.conv_batch_relu(x, self.num_filters[2])

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_2])
        x = self.conv_batch_relu(x, self.num_filters[1])
        x = self.conv_batch_relu(x, self.num_filters[1])

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_1])
        x = self.conv_batch_relu(x, self.num_filters[0])
        x = self.conv_batch_relu(x, self.num_filters[0])

        return x





