import cv2
import numpy as np
import os
import random
import shutil
import sys

class DataAugmentation:

    def __init__(self, png_file):
        # change to the directory containing Utah satellite images
        self.basepath = 'D:/acb/Stanford/CS230/cowc-everything/cowc/datasets/patch_sets/counting/COWC_Counting_Utah_AGRC/Utah_AGRC/train/'
        self.croppath = 'D:/acb/Stanford/CS230/cowc-everything/cowc/datasets/patch_sets/counting/COWC_Counting_Utah_AGRC/Utah_AGRC/train/cropped/'
        self.inp_file = self.basepath + png_file
        self.crop_file = self.croppath + png_file[:-4] + '_crop.png'

    def cropBorder(self, plt_flag=False, save_orig=True):
        """
        Crops the border off an image, assumes the left/right borders are same thickness and top/bottom borders are same thickness.
        :param plt_flag: Boolean governing plotting of cropped image.
        :param save_orig: Boolean governing whether the original image is saved or deleted.
        """
        img = self.readPng()
        nx, ny, nz = img.shape  # find number of pixels of image
        height_pct, width_pct = self.findBorder(img)  # find fraction of the image height/width that are border
        crop_img = img[round(width_pct*nx):round((1-width_pct)*nx)-1,round(height_pct*ny):round((1-height_pct)*ny)-1,:]  # crop the image
        cv2.imwrite(self.crop_file,crop_img)
        if plt_flag:
            self.plotImage(crop_img)
        if not save_orig:
            os.remove(self.inp_file)
        return

    def findBorder(self, img):
        """
        Calculates the percentage of the width & height of image that is the border.
        :param img: Image array with border.
        :return: Tuple of height and width border percentage.
        """
        nx, ny, nz = img.shape  # find number of pixels of image
        rgb_width_start = img[round(nx/2),0,:] # index of middle of left side of image
        rgb_height_start = img[round(ny/2),0,:] # index of middle of top side of image
        i = 1
        while True:  # search for end of left border
            rgb_width = img[round(nx/2),i,:]
            if not np.array_equal(rgb_width,rgb_width_start):
                width_pct = i / nx
                break
            i += 1
        i = 1
        while True:  # search for end of top border
            rgb_height = img[round(ny/2),i,:]
            if not np.array_equal(rgb_height,rgb_height_start):
                height_pct = i / ny
                break
            i += 1
        return height_pct, width_pct

    def plotImage(self, img):
        """
        Plot the image.
        :param img: Image array for plotting.
        """
        cv2.imshow('Image',img)
        cv2.waitKey(0)
        return

    def readPng(self):
        """
        Read in image file.
        """
        return cv2.imread(self.inp_file)


if __name__ == '__main__':
    N_crop = 100
    da = DataAugmentation('00.00003.04268.060.png')  # instantiate data augmentation class simply to get da.basepath (very sloppy)
    filenames = random.sample(os.listdir(da.basepath), N_crop)
    for fname in filenames:
        da = DataAugmentation(fname)  # instantiate data augmentation class
        da.cropBorder(save_orig=False)
