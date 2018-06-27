

KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "ops_logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants
CAT_DIR = "../Corrosion_Dataset/defects/"
LABEL_DIR = "../Corrosion_Dataset/labels/"
CAT1            = "rust"
CAT2            = "notrust"
CAT1_ONEHOT     = [1,0]
CAT2_ONEHOT     = [0,1]


LEARNING_RATE = 0.01               #Learning rate for training the CNN
CNN_LOCAL1 = 64                  #Number of features output for conv layer 1
CNN_LOCAL2 = 64                  #Number of features output for conv layer 1
CNN_GLOBAL1 = 64                  #Number of features output for conv layer 2
CNN_CLASSES      = 2
CNN_EPOCHS       = 1000
CNN_FULL1   = 500                #Number of features output for fully connected layer1
IMG_SIZE = 8                       #MAKE SURE THIS NUMBER IS AN EVEN NUMBER
IMG_DEPTH   = 3
BATCH_SIZE = 800
