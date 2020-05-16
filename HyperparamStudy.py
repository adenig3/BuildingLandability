import urllib.request
import json
import cv2
import math
import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from JSONManager import JSONManager
from ModelUtils import ModelUtils
from Models.UNet import UNet

class HyperparamStudy:

    def __init__(self, image_size=192, nn_type="UNet", n_layers=9, learning_rate=1.0, decay_steps=100000, decay_rate=0.96, lambd=0.0,
                 batch_size=64, epochs=100, optimizer="adam", loss="binary_crossentropy", plot_flag=False):
        self.image_size = image_size
        self.nn_type = nn_type
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.lambd = lambd
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.plot_flag = plot_flag
        self.model = None

        # most of those variables won't change, if they need to change, change by hand
        self.train_path = "train/"
        self.valid_path = "validate/"
        self.json_path ="export-2020-05-14T18_00_56.691Z.json"
        self.sets = ['main set 1']
        self.inputs_path = 'Inputs/'
        self.labels_path = 'Labels/'
        self.save_path = "Models/" + nn_type + "n" + str(n_layers) + "lr" + str(learning_rate) + "dr" + str(decay_rate) \
            + "lamb" + str(lambd) + "bs" + str(batch_size) + "e" + str(epochs)
        self.save_path = self.save_path.replace(".","_") + ".h5" # replaces decimal points with underscores to prevent mess-ups in file name
        self.load_path = "Models/" + nn_type + "n" + str(n_layers) + "lr" + str(learning_rate) + "dr" + str(decay_rate) \
            + "lamb" + str(lambd) + "bs" + str(batch_size) + "e" + str(epochs)
        self.load_path = self.load_path.replace(".","_") + ".h5" # replaces decimal points with underscores to prevent mess-ups in file name
        self.output_path = "Hyperparameters_Data.xlsx"


    def runner(self):
        JM = JSONManager(self.json_path, self.sets, self.inputs_path, self.labels_path, self.data_pct)
        MU = ModelUtils()

        x_train, y_train = JM.load_dataset(self.train_path+self.inputs_path, self.train_path+self.labels_path)
        x_val, y_val = JM.load_dataset(self.valid_path+self.inputs_path, self.valid_path+self.labels_path)
        x_train, y_train = JM.normalize_dataset(x_train,y_train)

        y_train2 = y_train[:, :, :, 0]
        y_train = np.expand_dims(y_train2, axis=-1)

        if self.nn_type == "UNet":
            num_filters = [2**(4+i) for i in range(math.ceil(self.n_layers/2))]  # number of filters in each layer is a power of 2 starting at 16 up to bottleneck
            self.model = UNet(num_filters, self.image_size)
        elif self.nn_type == "ResNet":
            #self.model = ResNet(, self.image_size)
        else:
            raise Exception("Invalid neural network architecture entered.")
        # ADD LEARNING RATE DECAY AND L2 REGULARIZATION OPTIONS TOMORROW MORNING
        self.model = self.model.configure()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["acc"])
        self.model.summary()

        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2)  # validation_data=[x_val,y_val]
        MU.save_model(self.model, self.save_path)

        if self.plot_flag:
            MU.show_validation(self.model,x_val, y_val)
            MU.show_train(self.model,x_train, y_train)
        return


    def write(self):
        # open self.output_path, if empty file, add column titles, go to new line, add accuracies & other info for current model
        return


if __name__ == 'main':
    # Fill these np.arrays with as many options as user wishes (should all be same length)
    image_size = np.array([])
    nn_type = []  # can use syntax ["UNet]*len(image_size) or ["UNet" for i in range(len(image_size))] to make a list of repeated nn architectures
    n_layers = np.array([])
    learning_rate = np.array([])
    decay_steps = np.array([])
    decay_rate = np.array([])
    lambd = np.array([])
    batch_size = np.array([])
    epochs = np.array([])
    optimizer = []
    loss = []
    plot_flag = False

    num_iter = image_size.size
    #num_iter = max([len(image_size),len(nn_type),len(n_layers),len(learning_rate),len(decay_steps),len(decay_rate),len(lambd),
                   # len(batch_size),len(epochs),len(optimizer),len(loss)])

    for i in range(num_iter):
        HPS = HyperparamStudy(image_size=image_size[i], nn_type=nn_type[i], n_layers=n_layers[i], learning_rate=learning_rate[i], decay_steps=decay_steps[i],
                          decay_rate=decay_rate[i], lambd=lambd[i], batch_size=batch_size[i], epochs=epochs[i], optimizer=optimizer[i],
                          loss=loss[i],plot_flag=plot_flag)
        HPS.runner()
        HPS.write()
        # Add capability to find each model's training and validation accuracy from model variable and write to an Excel file
        # so we can log the performance of each set of hyperparameters with a write() function in HyperparamStudy