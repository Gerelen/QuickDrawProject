from utils import proper_resize,one_hot_Y,rgb2gray
from unload_data import prepareData
from keras.models import load_model
import matplotlib.pyplot as plt
from draw import showImage
from PIL import Image
import numpy as np
import pickle
import PIL
import cv2


img = showImage() 
img = proper_resize(img)   
img = rgb2gray(img)  
img = img.reshape(-1,28,28,1)

model = load_model('test.model')
lb = pickle.loads(open('lb.pickle','rb').read())

pred = model.predict(img)[0]
idx = np.argmax(pred)
label = lb.classes_[idx]
print(label)
print(pred[idx] * 100)

