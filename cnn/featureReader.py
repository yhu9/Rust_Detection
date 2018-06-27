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

#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py

def cnn_readOneImg2(image_dir):
    inputs = []
    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    original,markers = getSegments(img,False)
    uniqueMarkers = np.unique(markers)
    canvas = original.copy()
    for uq_mark in uniqueMarkers:
        #make a canvas and paint each unique segment
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        canvas[markers == uq_mark] = [b,g,r]

    return(canvas,markers)

#function for stitching 4 different images together
#top left = cat1 = 'treematter'
#top right = cat2 = 'plywood'
#bot left = cat3 = 'cardboard'
#bot right = cat4 = 'construction'
def getStitchedImage():

    #initialize variablees
    cat1_dir = constants.cat1_dir
    cat2_dir = constants.cat2_dir
    cat3_dir = constants.cat3_dir
    cat4_dir = constants.cat4_dir
    cat1files = []
    cat2files = []
    cat3files = []
    cat4files = []

    #check if the file directories exist and push all files into their respective categories
    if os.path.exists(cat1_dir) and os.path.exists(cat2_dir) and os.path.exists(cat3_dir) and os.path.exists(cat4_dir):
        for filename in os.listdir(cat1_dir):
            cat1files.append(filename)
        for filename in os.listdir(cat2_dir):
            cat2files.append(filename)
        for filename in os.listdir(cat3_dir):
            cat3files.append(filename)
        for filename in os.listdir(cat4_dir):
            cat4files.append(filename)

    #pick a random file from the list of files for each category and read them in
    random.seed(None)
    a = random.randint(0,len(cat1files) - 1)
    b = random.randint(0,len(cat2files) - 1)
    c = random.randint(0,len(cat3files) - 1)
    d = random.randint(0,len(cat4files) - 1)
    img1 = cv2.imread(cat1_dir + '/' + cat1files[a],cv2.IMREAD_COLOR)
    img2 = cv2.imread(cat2_dir + '/' + cat2files[b],cv2.IMREAD_COLOR)
    img3 = cv2.imread(cat3_dir + '/' + cat3files[c],cv2.IMREAD_COLOR)
    img4 = cv2.imread(cat4_dir + '/' + cat4files[d],cv2.IMREAD_COLOR)

    #create the image by resizing and putting them into their correct positions
    topleft = cv2.resize(img1,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomleft = cv2.resize(img2,(500,500),interpolation = cv2.INTER_CUBIC)
    topright = cv2.resize(img3,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomright = cv2.resize(img4,(500,500),interpolation = cv2.INTER_CUBIC)
    toprow = np.concatenate((topleft,topright),axis = 1)
    bottomrow = np.concatenate((bottomleft,bottomright),axis = 1)
    full_img = np.concatenate((toprow,bottomrow),axis = 0)

    return full_img

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

        #turns out the reads aren't completely binary. We have a lot of gray areas so we convert it to binary
        gray_img = cv2.imread(labelfile,cv2.IMREAD_GRAYSCALE)
        thresh, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

        #create the training / testing set by picking a batch size of random points from the random image
        #even out the training between points that are and are not rust
        h,w,d = img.shape
        low = int(constants.IMG_SIZE / 2)
        offset = low
        high_w = int(w - (constants.IMG_SIZE / 2) - 1)
        high_h = int(h - (constants.IMG_SIZE / 2) - 1)
        tmp = mask[low:high_h,low:high_w]
        x1s,y1s = np.where(tmp == 0)
        x2s,y2s = np.where(tmp == 255)

        #cut off the edges so we don't choose pixels on the boundry
        for i in range(n):
            j = random.randint(0,9)
            if j % 2 == 0:
                pt_id = random.randint(0,len(x1s) - 1)
                x = x1s[pt_id]
                y = y1s[pt_id]
                labels.append([0.0])
            else:
                pt_id = random.randint(0,len(x2s) - 1)
                x = x2s[pt_id]
                y = y2s[pt_id]
                labels.append([1.0])

            x += offset
            y += offset

            box_low1 = int(x - int(constants.IMG_SIZE / 2) )
            box_low2 = int(y - int(constants.IMG_SIZE / 2) )
            box_high1 = int(x + int(constants.IMG_SIZE / 2))
            box_high2 = int(y + int(constants.IMG_SIZE / 2))

            box = img[box_low1:box_high1,box_low2:box_high2]
            inputs.append(box)

            if len(box) == 0:
                print("ERROR")
                quit()
    else:
        print("%s directory does not exist" % cat_dir)
        quit()

    tmp = list(zip(inputs,labels))
    random.shuffle(tmp)
    inputs,labels = zip(*tmp)

    return np.array(inputs),np.array(labels)


