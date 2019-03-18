import numpy as np
import os

def prepareData(dataset):
	X = []
	Y = []
	for file in os.listdir(dataset):
		datasets = np.load(os.path.join(dataset,file))
		for arr in datasets:
			labels,ext = os.path.splitext(file)
			Y.append(labels)
			X.append(arr)

	return X,Y

#train = np.load('dataset/full_numpy_bitmap_The Eiffel Tower.npy')
#train = train.reshape(train.shape[0],28,28,1)
#train = train.astype('float64') / 255.0

