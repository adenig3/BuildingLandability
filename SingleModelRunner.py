
import cv2
import numpy as np
from JSONManager import JSONManager
from ModelUtils import ModelUtils
from Models.UNet import UNet
from Models.SegNet import SegNet
import tensorflow as tf
from tensorflow import keras



# Data paths and sets info
train_path = "train/"
valid_path = "validate/"
json_path ="export-2020-05-15T13_40_03.223Z.json"
sets = ['main set 1']
inputs_path = 'Inputs/'
labels_path = 'Labels/'
save_path = 'Models/SegNet.h5'
load_path = 'Models/SegNet.h5'
#load_path = 'ModelFile.h5'


image_size = 192
epochs = 70
batch_size = None
data_pct = [0.8, 0.1, 0.1]  # percent of data for training, validation, and test

# General parameters
download = False
sort = False
make_model = True


JM = JSONManager(json_path, sets, inputs_path, labels_path, data_pct)
MU = ModelUtils()


if download:
    JM.download_training_set()
if sort:
    JM.sort_dataset()

x_train, y_train = JM.load_dataset(train_path+inputs_path, train_path+labels_path)
x_val, y_val = JM.load_dataset(valid_path+inputs_path, valid_path+labels_path)
x_train, y_train = JM.normalize_dataset(x_train,y_train)

if make_model:
    y_train2 = y_train[:, :, :, 0]
    y_val2 = y_val[:, :, :, 0]
    y_train = np.expand_dims(y_train2, axis=-1)
    y_val = np.expand_dims(y_val2, axis=-1)

    #model = UNet([16, 32, 64, 128, 256], image_size)
    #model = UNet([64, 128, 256, 512], image_size)
    #model = SegNet([64, 128, 256, 512], image_size)
    model = SegNet([16, 32, 64, 128, 256], image_size)
    model = model.configure()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2)  # validation_data=[x_val,y_val]
    MU.save_model(model, save_path)
else:
    model = MU.load_model(load_path)

MU.show_validation(model,x_val, y_val)
MU.show_train(model,x_train, y_train)


""" Future Ideas:
1) Integrate Batch Normalization into standard UNet
2) Look at using pre-trained weights for diff encoders
3) Look into regularization or drouput
4) Consider increasing  number of filters per layer for deeper conv blocks"""



