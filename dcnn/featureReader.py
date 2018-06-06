import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
from segmentModule import *
from matplotlib import pyplot as plt

#function for stitching 4 different images together
def testStitcher():
    for i in range(10):
        full_img = stitchImage()
        rgb = scipy.misc.toimage(full_img)
        cv2.imshow('stiched image',full_img)
        cv2.imwrite('full_img.png',full_img)
        cv2.waitKey(0)

#gets n patches from an image with its respective label
def getPixelBatch(n):
    inputs = []
    labels = []

    #initialize variablees
    cat_dir = constants.CAT_DIR
    label_dir = constants.LABEL_DIR

    categories = [constants.CAT1_ONEHOT,constants.CAT2_ONEHOT]
    images = []
    files = []

    #check if the file directories exists
    if os.path.exists(cat_dir) and os.path.exists(label_dir):
        rand_findex = random.randint(0,len(os.listdir(cat_dir)) - 1)

        #get a random image and label and read the images in
        catfile = os.path.join(cat_dir,os.listdir(cat_dir)[rand_findex])
        labelfile = os.path.join(label_dir,os.path.splitext(os.path.basename(catfile))[0] + 'gt' + os.path.splitext(os.path.basename(catfile))[1])
        img = cv2.imread(catfile,cv2.IMREAD_COLOR)
        gray_img = cv2.imread(labelfile,cv2.IMREAD_GRAYSCALE)
        thresh, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

        #create the training / testing set by picking a batch size of random points from the random image
        h,w,d = img.shape
        h2,w2 = mask.shape
        low = int(constants.IMG_SIZE / 2)
        high_w = int(w - (constants.IMG_SIZE / 2) - 1)
        high_h = int(h - (constants.IMG_SIZE / 2) - 1)
        for i in range(n):
            y = random.randint(low,high_h)
            x = random.randint(low,high_w)
            box_low1 = int(y - (constants.IMG_SIZE / 2))
            box_low2 = int(x - (constants.IMG_SIZE / 2))
            box_high1 = int(y + (constants.IMG_SIZE / 2))
            box_high2 = int(x + (constants.IMG_SIZE / 2))

            box = img[box_low1:box_high1,box_low2:box_high2]
            inputs.append(box)
            if np.all(mask[y,x] == 0):
                labels.append(constants.CAT1_ONEHOT)
            elif np.all(mask[y,x] == 255):
                labels.append(constants.CAT2_ONEHOT)
            else:
                print('error with the labels image. IT IS NOT WHITE AND BLACK!')
                print(mask[y,x])
                quit()
    else:
        print("%s directory does not exist" % cat_dir)
        quit()

    return inputs,labels

