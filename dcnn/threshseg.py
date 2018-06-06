import numpy as np
import cv2
import sys
import os
import random
import segmentModule as seg

MIN_DENSITY = 1000
nothreshflag = 'nothresh' in sys.argv
hsvsegflag = 'hsvseg' in sys.argv

TREEMATTER = [0,0,255]
PLYWOOD = [0,255,0]
CARDBOARD = [255,0,0]
BLACKBAG = [255,255,0]
TRASHBAG = [255,0,255]
BOTTLES = [0,255,255]

CATS=[TREEMATTER,PLYWOOD,CARDBOARD,BLACKBAG,TRASHBAG,BOTTLES]

def threshseg(original,mask,raw_values):
    #find out the shape of the raw values and make sure it is what we are expecting
    h,w,d = raw_values.shape
    print(original.shape)
    print(mask.shape)
    print(raw_values.shape)
    exit()
    if d == 3:
        print('sorry this is an image')
    elif d == 6:

        #threshold mask = max(raw) / sum(abs(raw))
        totals = abs(raw_values[:,:,0]) + abs(raw_values[:,:,1]) + abs(raw_values[:,:,2]) + abs(raw_values[:,:,3]) + abs(raw_values[:,:,4]) + abs(raw_values[:,:,5])
        maximums = np.amax(raw_values,axis=2)
        thresh_mask = maximums / totals

        #segment the image
        if hsvsegflag:
            hsvimg = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
            segmented_image, labels = seg.getSegments(hsvimg,md=MIN_DENSITY)
        else:
            segmented_image, labels = seg.getSegments(original,md=MIN_DENSITY)

        #create the segmented image with clear differing colors
        #create the segmented image with majority rule using classification categories
        unique_labels = np.unique(labels)
        blank1 = original - original
        blank2 = original - original
        for label in unique_labels:

            #randomly paint blank1
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
            blank1[ labels == label] = [b,g,r]

            #find majority category and paint blank2
            majority = -1
            for cat in CATS:
                count = np.count_nonzero(np.all(mask[labels == label] == cat,axis=2))
                if count > majority:
                    classification = cat
            blank2[labels == label] = cat

        return blank1,blank2

if __name__ == '__main__':

    if len(sys.argv) >= 4 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]) and os.path.exists(sys.argv[3]):
        original = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
        mask = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        raw_values = np.load(sys.argv[3])

        #get the segmentation
        ms_segmentation, majority_segmentation = threshseg(original,mask,raw_values)

        #write resulting images in the results results directory
        fout1 = os.path.join('results','meanshift_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        fout2 = os.path.join('results','majoritysegmentation_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        cv2.imwrite(fout1,ms_segmentation)
        cv2.imwrite(fout2,majority_segmentation)
