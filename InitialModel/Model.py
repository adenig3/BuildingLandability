import urllib.request
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 2 #Either Landable or Not landable
# Read In the JSON
def read_json(filename):
    with open(filename) as f:
        json_data = json.load(f)
    return json_data

#Shows an example image
def show_input_label(x, y, example_num):
    img_concat = np.concatenate((x[example_num], y[example_num]), axis=1)
    cv2.imshow('Input/Label ' + str(example_num), img_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Downloads the trainings st images from the URL in json file
def download_training_set_images(json_data, inputs_filepath, labels_filepath):
    i = 0
    for example in json_data:
        input_fullpath = inputs_filepath + 'Input' + str(i) + '.png'
        label_fullpath = labels_filepath + 'Label' + str(i) + '.png'
        img_url = example['Labeled Data']
        mask_url = example['Label']['objects'][0]['instanceURI'] #White is landable
        mask_type = example['Label']['objects'][0]['title']
        urllib.request.urlretrieve(img_url, input_fullpath) #Raw Input Image
        urllib.request.urlretrieve(mask_url, label_fullpath)
        if mask_type == "Not Landable Zone": #Invert the Image
            img = cv2.imread(label_fullpath)
            cv2.imwrite(label_fullpath, 255-img)
        i += 1

# Loads the training set as numpy arrays
def load_training_set(inputs_filepath, labels_filepath):
    input_filenames = os.listdir(inputs_filepath)
    label_filenames = os.listdir(labels_filepath)
    if len(input_filenames) != len(label_filenames):
        raise Exception("Input and Label Sizes are Not the Same")
    x_train = []
    y_train = []
    i = 0
    for i in range(len(input_filenames)):
        x_train.append(np.asarray(cv2.imread(inputs_filepath + 'Input' + str(i) + '.png')[0:192, 0:192, :]))
        y_train.append(np.asarray(cv2.imread(labels_filepath + 'Label' + str(i) + '.png')[0:192, 0:192, :]))
        i += 1
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train, y_train

#Normalizes the dataset for sigmoid
def normalize_dataset(x, y):
    return x/255, y/255

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



image_size = 192
train_path = "Inputs/"
epochs = 5
batch_size = 8



# General parameters
download = False
if download:
    json_data = read_json(filename="export-2020-05-07T01_04_15.454Z.json")
    download_training_set_images(json_data, 'Inputs/', 'Labels/')

x_train, y_train = load_training_set('Inputs/', 'Labels/')
x_train, y_train = normalize_dataset(x_train,y_train)
y_train2 = y_train[:,:,:,0]
y_train = np.expand_dims(y_train2, axis=-1)


model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

model.fit(x=x_train, y=y_train, validation_split= 0.1, batch_size=None, epochs=10,verbose=2,validation_data=None)

#show_input_label(x_train, y_train, 155)

# Implement dataGen class






