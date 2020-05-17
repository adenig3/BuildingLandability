from tensorflow import keras
import cv2
import numpy as np
# ASSUMES KERAS MODEL DATATYPE
class ModelUtils:
    def save_model(self, model, save_path):
        model.save(save_path)

    def load_model(self,  load_path):
        return keras.models.load_model(load_path, compile=False)

    def show_validation(self, model, x_val, y_val):
        for ind in range(x_val.shape[0]):
            # ind = int(input('Number: '))
            result = model.predict(x_val[ind:ind + 1, :, :, :])
            truth = y_val[ind]
            thresh, result = cv2.threshold(result[0], 0.50, 255, cv2.THRESH_BINARY)
            grey_line = np.zeros((truth.shape[0], 2))
            grey_line[:, :, ] = 128
            img_concat = np.concatenate((result, np.concatenate((grey_line, truth[:, :, 0]), axis=1)), axis=1)
            cv2.imshow('Validation: Prediction vs. Truth', img_concat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_train(self, model, x_train, y_train):
        for ind in range(x_train.shape[0]):
            # ind = int(input('Number: '))
            result = model.predict(x_train[ind:ind + 1, :, :, :])
            truth = y_train[ind]
            thresh, result = cv2.threshold(result[0], 0.50, 255, cv2.THRESH_BINARY)
            grey_line = np.zeros((truth.shape[0], 2))
            grey_line[:, :, ] = 128
            img_concat = np.concatenate((result, np.concatenate((grey_line, truth[:, :, 0]), axis=1)), axis=1)
            cv2.imshow('Train: Prediction vs. Truth', img_concat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()