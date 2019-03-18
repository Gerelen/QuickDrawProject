from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from skimage import color
from PIL import Image
import numpy as np
import PIL

def one_hot_Y(Y):
	global lb
	lb = LabelBinarizer()
	label = np.array(Y)
	labels = lb.fit_transform(label)
	labels = to_categorical(labels)

	return labels, lb

def returnTrainTest(X,Y,test_size=0.1):

	Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=test_size,random_state=42)

	Xtrain = Xtrain.astype('uint8') / 255.0
	Xtest = Xtest.astype('uint8') / 255.0

	return (Xtrain,Ytrain),(Xtest,Ytest)

def reshapeX(X):
	X = np.array(X)
	return X.reshape(X.shape[0],28,28,1)

# Properly resize image from draw.py to a 28x28.
def proper_resize(img):
	baseheight = 28
	basewidth = 28
	img = Image.fromarray(img, 'RGB')#change img to gray
	hpercent = (baseheight / float(img.size[1]))
	wsize = int((float(img.size[0]) * float(hpercent)))
	img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
	img = np.array(img)
	return img

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])