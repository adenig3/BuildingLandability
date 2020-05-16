import urllib.request
import json
from JSONManager import JSONManager
import cv2
import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow_examples.models.pix2pix import pix2pix


OUTPUT_CHANNELS = 2 #Either Landable or Not landable

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


#Make the model. TBH idk what's happening
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


#Save the model
def save_model(model_to_save, filename):
    model_to_save.save(filename)


#Load the model
def load_model(filename):
    return keras.models.load_model(filename)




image_size = 192
train_path = "train/"
epochs = 5
batch_size = 8
data_pct = [0.8,0.1,0.1]  # percent of data for training, validation, and test


# General parameters
download = False
sort = False
make_model = False
json_runner = JSONManager(json_path='export-2020-05-07T01_04_15.454Z.json', sets_to_include='main set 1',
                          inputs_path='Inputs/', labels_path='Labels/', data_pct=data_pct)

if download:
    json_data = json_runner.read_json()
    json_runner.download_training_set()

if sort:
    json_runner.sort_dataset()
x_train, y_train = json_runner.load_dataset('train/Inputs/', 'train/Labels/')
x_val, y_val = json_runner.load_dataset('validate/Inputs/', 'validate/Labels/')
x_train, y_train = json_runner.normalize_dataset(x_train,y_train)
# json_runner.show_input_label(x_train, y_train, 155)

if make_model:
    y_train2 = y_train[:,:,:,0]
    y_train = np.expand_dims(y_train2, axis=-1)


    model = UNet()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    model.fit(x=x_train, y=y_train, batch_size=None, epochs=200,verbose=2)  # validation_data=[x_val,y_val]
    save_model(model, 'ModelFile.h5')


model = load_model('ModelFile.h5')

for ind in range(x_val.shape[0]):
    #ind = int(input('Number: '))
    result = model.predict(x_val[ind:ind+1,:,:,:])
    truth = y_val[ind]
    thresh, result = cv2.threshold(result[0], 0.50, 255, cv2.THRESH_BINARY)
    grey_line = np.zeros((image_size,2))
    grey_line[:,:,] = 128
    img_concat = np.concatenate((result,np.concatenate((grey_line,truth[:,:,0]), axis=1)), axis=1)
    cv2.imshow('Prediction vs. Truth', img_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for ind in range(x_train.shape[0]):
    #ind = int(input('Number: '))
    result = model.predict(x_train[ind:ind+1,:,:,:])
    truth = y_train[ind]
    thresh, result = cv2.threshold(result[0], 0.50, 255, cv2.THRESH_BINARY)
    grey_line = np.zeros((image_size,2))
    grey_line[:,:,] = 128
    img_concat = np.concatenate((result,np.concatenate((grey_line,truth[:,:,0]), axis=1)), axis=1)
    cv2.imshow('Prediction vs. Truth', img_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













