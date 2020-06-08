
import cv2
import numpy as np
from JSONManager import JSONManager
from ModelUtils import ModelUtils
from Models.UNet import UNet
from DataAugmentation import DataAugmentation
from Models.VGG_UNet import VGG_UNet
import tensorflow as tf
from tensorflow import keras
import pickle



# Data paths and sets info
train_path = "train/"
valid_path = "validate/"
json_path ="export-2020-05-18T16_04_41.148Z.json"
sets = ['first100train', 'main set 1']
inputs_path = 'Inputs/'
labels_path = 'Labels/'
load_path = 'Models/UNetn9nf16_0-32_0-64_0-128_0-256_0lr0_001dordown0_0dorup0_0dr0lamb0bs32e70.h5'
save_path = 'Models/UNetn9nf16_0-32_0-64_0-128_0-256_0lr0_001dordown0_0dorup0_0dr0lamb0bs32e70.h5'



image_size = 192
epochs = 20
batch_size = 16
data_pct = [0.95, 0.025, 0.025]  # percent of data for training, validation, and test

# General parameters
download = False
sort = False
make_pickle = False
augment = False
make_model = False


JM = JSONManager(json_path, sets, inputs_path, labels_path, data_pct)
MU = ModelUtils()
DA = DataAugmentation('Doesntmatter.png', 'D:/acb/GitHub/BuildingLandability/train')

if download:
    JM.download_training_set()

if sort:
    JM.sort_dataset()

if augment:
    DA.flipAll(True, True)

x_train, y_train = JM.load_dataset(train_path+inputs_path, train_path+labels_path)
x_val, y_val = JM.load_dataset(valid_path+inputs_path, valid_path+labels_path)
#Invert dataset to test something
#y_train = np.invert(y_train)
#y_val = np.invert(y_val)
x_train, y_train = JM.normalize_dataset(x_train, y_train)
y_train2 = y_train[:, :, :, 0]
y_val2 = y_val[:, :, :, 0]
y_train = np.expand_dims(y_train2, axis=-1)
y_val = np.expand_dims(y_val2, axis=-1)
# if make_pickle:
#     x_train, y_train = JM.load_dataset(train_path+inputs_path, train_path+labels_path)
#     x_val, y_val = JM.load_dataset(valid_path+inputs_path, valid_path+labels_path)
#     #Invert dataset to test something
#     #y_train = np.invert(y_train)
#     #y_val = np.invert(y_val)
#     x_train, y_train = JM.normalize_dataset(x_train, y_train)
#     y_train2 = y_train[:, :, :, 0]
#     y_val2 = y_val[:, :, :, 0]
#     y_train = np.expand_dims(y_train2, axis=-1)
#     y_val = np.expand_dims(y_val2, axis=-1)
#     np.save('x_train.npy',x_train)
#     np.save('y_train.npy',y_train)
#     np.save('x_val.npy',x_val)
#     np.save('y_val.npy',y_val)
    # with open('train.pickle', 'wb') as f:
    #     pickle.dump([x_train, y_train], f, protocol=4)
    # with open('val.pickle', 'wb') as f:
    #     pickle.dump([x_val, y_val], f, protocol=4)

# with open('train.pickle', 'rb') as f:
#     x_train, y_train = pickle.load(f)
# with open('val.pickle', 'rb') as f:
#     x_val, y_val = pickle.load(f)

if make_model:

    #model = UNet([16, 32, 64, 128, 256], image_size, 0, [0.6, 0.05], [False, False])
    model = UNet([32, 64, 128, 256, 512], image_size, 0, [0.6, 0.0], [False, False])
    for i in range(0,5):
        if i == 0:
            model = model.configure()
            #model = MU.load_model(save_path)
        else:
            model = MU.load_model(save_path)
        #model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
        #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=['acc', tf.keras.metrics.MeanIoU(num_classes=2)])
        model.summary()
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2)  # validation_data=[x_val,y_val]
        MU.save_model(model, save_path)
else:

    y_train2 = y_train[:, :, :, 0]
    y_val2 = y_val[:, :, :, 0]
    y_train = np.expand_dims(y_train2, axis=-1)
    y_val = np.expand_dims(y_val2, axis=-1)
    model = MU.load_model(load_path)
    #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()
    preds = model.evaluate(x_val, y_val)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

MU.show_validation(model,x_val, y_val)
MU.show_train(model,x_train, y_train)


""" Future Ideas:
1) Integrate Batch Normalization into standard UNet
2) Look at using pre-trained weights for diff encoders (imagenet weights) and transfer learning the decoder
3) Consider penalizing a false positive as more (when it says land but you cant
4) Consider increasing  number of filters per layer for deeper conv blocks"""

"""Comments
1) Regularization of 0.0001 is pretty solid for UNet
2) Similar results are obtained with results of 0.7. 0.8 too much
3) Maybe we don't want dropout on the upsampling segment?
4) Dropout of 0.7 too much for down and up dropout layers"""

"""Problems:
1) Its can identify prominent edges, but doesnt know whether it's large enough, or if road/texture is appropriate"""
