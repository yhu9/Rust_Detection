#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import math
import sys
import os

#Python Modules
import constants
import featureReader

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################

####################################################################################################################################
#Helper Functions
####################################################################################################################################

####################################################################################################################################
#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if not os.path.exists('log'):
        os.makedirs('log')

    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        y = tf.placeholder('float',[None,constants.CNN_CLASSES])
        weights = {}
        biases = {}

        #local convolution pathway
        weights['W_local1'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_local1'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv1 = tf.nn.conv2d(x,weights['W_local1'],strides=[1,1,1,1],padding='SAME',name='local1')
        local1 = tf.nn.relu(conv1 + biases['b_local1'])

        weights['W_local2'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL2]))
        biases['b_local2'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL2]))
        conv2 = tf.nn.conv2d(local1,weights['W_local2'],strides=[1,1,1,1],padding='SAME',name='local2')
        activations = tf.nn.relu(conv2 + biases['b_local2'])

        #create our first fully connected layer
        #magic number = width * height * n_convout
        magic_number = int(constants.IMG_SIZE * constants.IMG_SIZE * constants.CNN_LOCAL2)

        #fully conntected layer
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc'] = tf.Variable(tf.random_normal([magic_number,constants.CNN_FULL1]))
                biases['b_fc'] = tf.Variable(tf.random_normal([constants.CNN_FULL1]))
                layer1_input = tf.reshape(activations,[-1,magic_number])
                fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
                #fullyConnected = tf.nn.dropout(fullyConnected,constants.KEEP_RATE)
            tf.summary.histogram('activations_3',fullyConnected)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([constants.CNN_FULL1,constants.CNN_CLASSES]))
            biases['out'] = tf.Variable(tf.random_normal([constants.CNN_CLASSES]))
            predictions = tf.matmul(fullyConnected,weights['out'])+biases['out']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
            correct_prediction = tf.cast(correct_prediction,tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        #prediction operation
        predict_op = tf.argmax(predictions,1)
        tf.summary.scalar('accuracy',accuracy)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#helper functions

        def outputResults(image,mask,fout='segmentation.png'):
            #create the segmented image
            canvas = image.copy()
            canvas[mask == -1] = [0,0,0]
            canvas[mask == 0] = [0,0,0]
            canvas[mask == 1] = [255,255,255]

            #show the original image and the segmented image and then save the results
            cv2.imwrite(fout,canvas)

            #count the percentage of each category
            cat1_count = np.count_nonzero(mask == 0)
            cat2_count = np.count_nonzero(mask == 1)
            total = cat1_count + cat2_count

            #get the percentage of each category
            p1 = cat1_count / total
            p2 = cat2_count / total

            #output to text file
            with open('results.txt','a') as f:
                f.write("\nusing model: %s\n" % sys.argv[3])
                f.write("evaluate image: %s\n\n" % sys.argv[2])
                f.write("--------------------------------------------------------------------------------------\n")
                f.write("%s : %f\n" % (constants.CAT1,p1))
                f.write("%s : %f\n" % (constants.CAT2,p2))
                f.write("--------------------------------------------------------------------------------------\n")
                f.write("------------------------------------END-----------------------------------------------\n")
                f.write("--------------------------------------------------------------------------------------\n")

                greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

                #f.write out to the terminal what the most common category was for the image
                if(greatest == cat1_count):
                    f.write("\nthe most common category is: " + constants.CAT1)
                elif(greatest == cat2_count):
                    f.write("\nthe most common category is: " + constants.CAT2)
                else:
                    f.write("\nsorry something went wrong counting the predictions")

        #training mode trained on the image
        if(sys.argv[1] == 'train'):
            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                sess.run(init)
                merged = tf.summary.merge_all()

                #train the model
                acc = 0.00;
                modelpath = "model"
                logdir = 'log/traininglog.txt'
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                if not os.path.exists('log'):
                    os.makedirs('log')

                for epoch in range(constants.CNN_EPOCHS):

                    #get an image batch
                    batch_x,batch_y = featureReader.getPixelBatch(constants.BATCH_SIZE)

                    optimizer.run(feed_dict={x: batch_x, y: batch_y})

                    #evaluate the model using a test set
                    if epoch % 1 == 0:
                        eval_x,eval_y = featureReader.getPixelBatch(constants.BATCH_SIZE)
                        accnew = accuracy.eval({x: eval_x, y: eval_y})

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,'./model/cnn_model.ckpt')
                            print("highest accuracy found! model saved")

                        print('epoch: ' + str(epoch) + '     ' +
                                'accuracy: ' + str(accnew))
                        with open(logdir,'a') as log_out:
                            log_out.write('epoch: ' + str(epoch) + '     ' + 'accuracy: ' + str(accnew) + '\n')

        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            if os.path.isfile(sys.argv[2]):
                tmp = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                image = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #we recreate the image by painting the best_guess mask on a blank canvas with the same shape as image
                #initialize counters and the height and width of the image being tested.
                #constants.IMG_SIZE is the img size the learned model uses for classifiying a pixel.
                #NOT THE actual size of the image being tested
                h,w = image.shape[:2]
                count = 0
                count2 = 0
                best_guess = np.full((h,w),-1)
                raw_guess = np.full((h,w,6),0)
                tmp = []
                i0 = int(constants.IMG_SIZE / 2)
                j0 = int(constants.IMG_SIZE / 2)

                #define our log file and pixel segmentation file name
                if not os.path.exists('results'):
                    os.mkdir('results')
                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                logname = "results/rawoutput_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + '_' + modelname + ".txt"
                seg_file = 'results/' + os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"

                #GO THROUGH EACH PIXEL WITHOUT THE EDGES SINCE WE NEED TO MAKE SURE EVERY PART OF THE PIXEL AREA
                #BEING SENT TO THE MODEL IS PART OF THE IMAGE
                for i in range(int(constants.IMG_SIZE / 2),int(len(image) - (constants.IMG_SIZE / 2))):
                    for j in range(int(constants.IMG_SIZE / 2),int(len(image[0]) - (constants.IMG_SIZE / 2))):

                        #get the bounding box around the pixel to send to the training
                        box = image[i-int(constants.IMG_SIZE / 2):i+int(constants.IMG_SIZE / 2),j-int(constants.IMG_SIZE / 2):j+int(constants.IMG_SIZE / 2)]

                        #append the box to a temporary array
                        tmp.append(box)

                        #once the temporary array is the same size as the batch size, run the testing on the batch
                        if(len(tmp) == constants.BATCH_SIZE or count == ((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE)) - 1):
                            batch = np.array(tmp)
                            rawpredictions = predictions.eval({x:batch})
                            mask = rawpredictions.argmax(axis=1)

                            #now we go through the mask and insert the values to the correct position of best_guess which is a copy of
                            #the original image except all the values are -1
                            for raw,cat in zip(rawpredictions,mask):
                                best_guess[i0,j0] = cat
                                raw_guess[i0,j0] = raw
                                if j0 == (w - int(constants.IMG_SIZE/2)) - 1:
                                    j0 = int(constants.IMG_SIZE / 2)
                                    i0 += 1
                                else:
                                    j0 += 1

                            #give console output to show progress
                            outputResults(image,np.array(best_guess),fout=seg_file)
                            print('%i out of %i complete' % (count2,math.ceil(int((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE) / constants.BATCH_SIZE))))
                            #empty tmporary array
                            tmp = []
                            count2 += 1
                        count += 1

                np.save(logname,raw_guess)
        else:
            print("train ")
            print("trainseg ")
            print("test [image_filepath] [model_filepath]")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train)")

if __name__ == "__main__":
    tf.app.run()
