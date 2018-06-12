import cv2
import numpy as np
import os
import sys

def dice(img,gt,fout='dice_output.txt',writemode='w'):
    with open(fout,writemode) as fo:
        mask = 0
        TP = float(np.count_nonzero(np.logical_and(img == mask, gt == mask)))
        TN = float(np.count_nonzero(np.logical_and(np.logical_not(img == mask), np.logical_not(gt == mask))))
        FP = float(np.count_nonzero(np.logical_and(img == mask,np.logical_not(gt == mask))))
        FN = float(np.count_nonzero(np.logical_and(np.logical_not(img == mask),gt == mask)))
        P = float(TP + FN)
        N = float(TN + FP)

        PREC = (TP) / (TP + FP)
        ACC = (TP + TN) / (P + N)
        SENS= TP / P
        SPEC= TN / N

        DICE = TP / (P + FP)

        print('-----category: RUST ------')
        print('True Positive: %f' % TP)
        print('True Negative: %f' % TN)
        print('False Positive: %f' % FP)
        print('False Negative: %f' % FN)
        print('Positive: %f' % P)
        print('Negative: %f\n' % N)
        print('PRECICSION: %f' % PREC)
        print('SENSITIVITY: %f' % SENS)
        print('SPECIFICITY: %f' % SPEC)
        print('ACCURACY: %f' % ACC)
        print('DICE: %f' % DICE)
        print('--------------')

        fo.write('--------RUST--------\n\n\n')
        fo.write('True Positive: %f\n' % TP)
        fo.write('True Negative: %f\n' % TN)
        fo.write('False Positive: %f\n' % FP)
        fo.write('False Negative: %f\n' % FN)
        fo.write('Positive: %f\n' % P)
        fo.write('Negative: %f\n\n' % N)
        fo.write('PRECICSION: %f\n' % PREC)
        fo.write('SENSITIVITY: %f\n' % SENS)
        fo.write('SPECIFICITY: %f\n' % SPEC)
        fo.write('ACCURACY: %f\n' % ACC)
        fo.write('DICE: %f\n\n\n' % DICE)

        fo.write('---------------------------------------------------\n')

    return ACC,DICE

#mode of operation
if __name__ == "__main__":
    if len(sys.argv) == 3:
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]

        #check file existence
        if os.path.exists(dir1) and os.path.exists(dir2):
            #read the image
            #turns out the reads aren't completely binary. We have a lot of gray areas so we convert it to binary
            img = cv2.imread(dir1,cv2.IMREAD_GRAYSCALE)
            gray_img = cv2.imread(dir2,cv2.IMREAD_GRAYSCALE)
            thresh, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
            thresh, segImg = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

            dice(segImg[28:-28,28:-28],mask[28:-28,28:-28])

            #create results directory
            if not os.path.exists('results'):
                os.makedirs('results')
        else:
            sys.exit()
    else:
        print("wrong number of arguments")
        print("expecting 2 [segmented image file] [mask image file]")

