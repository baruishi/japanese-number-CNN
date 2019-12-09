# imports

import cv2
import numpy as np
import  os
from random import  shuffle
from tqdm import  tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


#training data  parameters
TRAIN_DIR = '/Users/kanaishibashi/Documents/Baru/python/japanese-number-CNN/images/training'
TEST_DIR = '/Users/kanaishibashi/Documents/Baru/python/japanese-number-CNN/images/test'
IMG_SIZE = 150
LR = 1e-3

#best results were achieved by 2conv
#90/10 triainingv/validation
#6conb = 0.69
#6conv = 0.715
#5conv = 0.
#4conv = 0.
#3conv = 0.
#2conv = 0.514


MODEL_NAME = 'japnumbers-{}-{}.model'.format(LR, '8conv-basic')


def label_img(img):
        word_label = img.split('.')[-3]
        #print(word_label)
        if word_label == '1': return [1, 0]
        elif word_label == '2': return [0, 1]


def create_train_data():
        training_data = []
        for img in tqdm(os.listdir(TRAIN_DIR)):
                try:
                        label = label_img(img)
                except IndexError:
                        #print('ERROR')
                        #print(img)
                        continue

                label = label_img(img)
                path = os.path.join(TRAIN_DIR, img)
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        print('saving data')
        return training_data


#create_train_data()

def process_test_data():
        testing_data = []
        for img in tqdm(os.listdir(TEST_DIR)):
            path = os.path.join(TEST_DIR, img)
            try:
                    img_num = img.split('.')[-3]
                    print('correct')
            except IndexError:
                    #print('ERROR')
                    #print(img)
                    continue
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        np.save('test_data.npy', testing_data)
        return  testing_data

#process_test_data()

#train_data = np.load('train_data.npy')
train_data = create_train_data()

#input layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
#originial dropout = 0.8
convnet = dropout(convnet, 0.8)

#number of output types
convnet = fully_connected(convnet, 2, activation='softmax')

#learning rate
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

#easy on mac/linux, so there might not be a need for tensorboard_dir
model = tflearn.DNN(convnet, tensorboard_dir='log')



#spreads the data
train = train_data[:-15]
test = train_data[-15:]

#this is what is getting fit - training data
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]


#this is for testing accuracy
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


#there is not "MODEL_NAME" in stead of "run_id=numbers
model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



print('end of prep_model')


