import cv2,os,sys
import numpy as np
from tqdm import tqdm
data_dir = 'dataset/'
directories = [i for i in os.listdir(data_dir)]
data = []
for i in tqdm(os.listdir(data_dir)):
    path = data_dir+i
    class_ = directories.index(i)
    for j in os.listdir(path):
        img = cv2.imread(os.path.join(path,j),0)
        img = cv2.resize(img , (64,64))
        data.append([img , class_])
np.save('Data.npy',data)


