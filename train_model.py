import numpy as np
from alexnet import alexnet
from tensorflow.keras.callbacks import TensorBoard
from random import shuffle
import cv2
from collections import Counter

WIDTH = 64
HEIGHT = 64

MODEL_NAME = 'Alexnet_v1.model'
tensorboard = TensorBoard(log_dir = 'Alexnet_v1_logs')
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
#print(set(test_y))

#model.fit(X, Y, batch_size=64, epochs=20 ,validation_data = (test_x,test_y),callbacks = [tensorboard])
#model.save(MODEL_NAME)














