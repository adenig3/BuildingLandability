import urllib.request
import json
import cv2
import numpy as np
import os
import random
import shutil


class JSONManager:
    def __init__(self, json_path, sets_to_include, inputs_path, labels_path, data_pct=[0.9,0.05,0.05]):
        self.json_path = json_path
        self.sets_to_include = sets_to_include
        self.inputs_path = inputs_path
        self.labels_path = labels_path
        self.data_pct = data_pct

    def read_json(self):
        with open(self.json_path) as f:
            json_data = json.load(f)
        return json_data

    def download_training_set(self):
        if not os.path.isdir(self.inputs_path):
            os.mkdir(self.inputs_path)
        if not os.path.isdir(self.labels_path):
            os.mkdir(self.labels_path)
        i = 0
        json_data = self.read_json()
        json_data = json_data[2527:] #Delete 372
        i = 2527
        for example in json_data:
            input_fullpath = self.inputs_path + 'Input' + str(i) + '.png'
            label_fullpath = self.labels_path + 'Label' + str(i) + '.png'
            img_url = example['Labeled Data']
            try: #In case there is no label on an image
                mask_url = example['Label']['objects'][0]['instanceURI']  # White is landable
                mask_type = example['Label']['objects'][0]['title']
            except:
                i += 1
                continue
            if example['Dataset Name'] in self.sets_to_include: #Include dataset names
                urllib.request.urlretrieve(img_url, input_fullpath)  # Raw Input Image
                urllib.request.urlretrieve(mask_url, label_fullpath)
                if mask_type == "Not Landable Zone":  # Invert the Image
                    img = cv2.imread(label_fullpath)
                    cv2.imwrite(label_fullpath, 255 - img)
                i += 1
        return

    def sort_dataset(self):
        input_filenames = os.listdir(self.inputs_path)
        label_filenames = os.listdir(self.labels_path)
        if len(input_filenames) != len(label_filenames):
            raise Exception("Input and Label sizes are Not the Same")
        if not os.path.isdir('train'):
            os.mkdir('train')
        if not os.path.isdir('train/Inputs/'):
            os.mkdir('train/Inputs/')
        if not os.path.isdir('train/Labels/'):
            os.mkdir('train/Labels/')
        if not os.path.isdir('validate'):
            os.mkdir('validate')
        if not os.path.isdir('validate/Inputs/'):
            os.mkdir('validate/Inputs/')
        if not os.path.isdir('validate/Labels/'):
            os.mkdir('validate/Labels/')
        if not os.path.isdir('test'):
            os.mkdir('test')
        if not os.path.isdir('test/Inputs/'):
            os.mkdir('test/Inputs/')
        if not os.path.isdir('test/Labels/'):
            os.mkdir('test/Labels/')
        num_data = len(input_filenames)  # total number of images in input data
        data_breakdown = [round(pct * num_data) for pct in self.data_pct]
        if sum(data_breakdown) != num_data:
            diff = num_data - sum(data_breakdown)
            data_breakdown[0] += diff
        i = 1
        for num in data_breakdown:
            inputs_random = random.sample(os.listdir(self.inputs_path), num)
            if i == 1:
                dest = os.path.join(os.getcwd(), 'train/')
            elif i == 2:
                dest = os.path.join(os.getcwd(), 'validate/')
            else:
                dest = os.path.join(os.getcwd(), 'test/')
            for file_input in inputs_random:
                file_label = 'Label' + file_input[len('Input'):]
                src_input = os.path.join(os.path.join(os.getcwd(), self.inputs_path), file_input)
                src_label = os.path.join(os.path.join(os.getcwd(), self.labels_path), file_label)
                dest_input = os.path.join(os.path.join(dest, 'Inputs/'), file_input)
                dest_label = os.path.join(os.path.join(dest, 'Labels/'), file_label)
                if os.path.isfile(src_input):
                    shutil.move(src_input, dest_input)  # move input into appropriate folder
                if os.path.isfile(src_label):
                    shutil.move(src_label, dest_label)  # move label into appropriate folder
            i += 1
        os.rmdir(self.inputs_path)
        os.rmdir(self.labels_path)
        return

    def load_dataset(self, inputs_path, labels_path):
        input_filenames = os.listdir(inputs_path)
        label_filenames = os.listdir(labels_path)
        if len(input_filenames) != len(label_filenames):
            raise Exception("Input and Label sizes are Not the Same")
        x = []
        y = []
        idx_list = []
        for file in input_filenames:
            file_num = file[5:-4]
            idx_list.append(int(file_num))
        for i in idx_list:
            x.append(np.asarray(cv2.imread(inputs_path + 'Input' + str(i) + '.png')[0:192, 0:192, :]))
            y.append(np.asarray(cv2.imread(labels_path + 'Label' + str(i) + '.png')[0:192, 0:192, :]))
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def normalize_dataset(self,x,y):
        return x/255, y/255

    def show_input_label(self, x, y, example_num):
        img_concat = np.concatenate((x[example_num], y[example_num]), axis=1)
        cv2.imshow('Input/Label ' + str(example_num), img_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


