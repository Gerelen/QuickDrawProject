from unload_data import prepareData
from utils import one_hot_Y, returnTrainTest,reshapeX
import matplotlib.pyplot as plt
import numpy as np
from model_design import createModel

# Load Data, X,Y
X,Y = prepareData('dataset/')


# Label Binarizer, to_categorical 'Y', also return classes from LabelBinarizer
Y,classes = one_hot_Y(Y)


# X to np_array, then reshape as MNIST (28,28,1), 1D array
X = reshapeX(X)

#Split X,Y into (Xtrain,Ytrain),(Xtest,Ytest), also change Xtrain,Xtest to 'float64' and / 255.0
(Xtrain,Ytrain),(Xtest,Ytest) = returnTrainTest(X,Y)


#Build model, return model, train_batch,val_batch, learning_rate_reduction
model,datagen,learning_rate_reduction = createModel(2)

#Acc: 98.40ish, Val_acc: 98.89 - 99.00
model.fit_generator(datagen.flow(Xtrain,Ytrain, batch_size=128),
					validation_data=(Xtest,Ytest),
					steps_per_epoch= len(Xtrain) // 128,
					epochs=1, verbose=1,shuffle=True)

model.save('test2.model')

