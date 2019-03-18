from keras.models import Sequential, Model, load_model
from keras.layers import (Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization,
						  Input,ZeroPadding2D,Activation,MaxPooling2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

def createModel(num_classes):

	model = Sequential()

	model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	datagen = ImageDataGenerator(
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.1,
        height_shift_range = 0.1,)

	#train_batch = datagen.flow(X, Y, batch_size = 128)
	#val_batch = datagen.flow(Xtest, Ytest, batch_size = 128)

	learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001)

	return model,datagen,learning_rate_reduction
