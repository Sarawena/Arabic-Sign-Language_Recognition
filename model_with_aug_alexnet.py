from alexnet import alexnet
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
import pickle
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import TensorBoard
from random import shuffle

WIDTH = 64
HEIGHT = 64

MODEL_NAME = 'Alexnet_v2.model'
tensorboard = TensorBoard(log_dir = 'Alexnet_v2_logs')
train_data = np.load('Data.npy', allow_pickle=True)
shuffle(train_data)


train = train_data[:-1000]
val = train_data[-1000:]


X = np.array([i[0] for i in train])
print(X.shape)

X = X.reshape(-1,WIDTH,HEIGHT,1)

print(X.shape)
Y = [i[1] for i in train]



shape = X.shape[1:]
output_shape = 32
model = alexnet(shape,output_shape)

X = np.array(X)
Y = np.array(Y)

test_x = np.array([i[0] for i in val]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in val]

test_x = np.array(test_x)
test_y = np.array(test_y)



trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False)
valAug = ImageDataGenerator()

h = model.fit_generator(trainAug.flow(X, Y),steps_per_epoch=len(X) // 32,validation_data=valAug.flow(test_x, test_y),validation_steps=len(test_x // 32),epochs=10,callbacks = [tensorboard])
model.save(MODEL_NAME)






#print(set(test_y))

#model.fit(X, Y, batch_size=64, epochs=20 ,validation_data = (test_x,test_y),callbacks = [tensorboard])
#model.save(MODEL_NAME)














