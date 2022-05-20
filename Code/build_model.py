# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:14:27 2022

@author: Gerges_Hanna
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import numpy as np
import build_model
from keras.wrappers.scikit_learn import KerasClassifier



# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 


def CNN_model(input_shape=(128,128,3),output_len=10,n_hidden=2,n_neurons=100,activation=tf.keras.activations.relu,optimizer=tf.keras.optimizers.Adam,learning_rate=None,add_batch_normalize=True):

    model=Sequential()
    #Input shape
    model.add(keras.Input(shape=input_shape,name="input_layer"))
    # ===========================First Block==================================================
    #adding convolution layer
    model.add(Conv2D(32,(3,3),activation=activation,padding="same",name='first_Conv2D'))
    #adding pooling layer
    model.add(MaxPooling2D((2,2),name='first_MaxPooling2D'))
    # ==================================================================================
    
    # ===========================Second Block==================================================
    #adding convolution layer
    model.add(Conv2D(64,(3,3),activation=activation,padding="same",name='second_Conv2D'))
    #adding pooling layer
    model.add(MaxPooling2D((2,2),name='second_MaxPooling2D'))
    # ==================================================================================
    
    # ===========================Third Block==================================================
    #adding convolution layer
    model.add(Conv2D(128,(3,3),activation=activation,padding="same",name='third_Conv2D'))
    #adding pooling layer
    model.add(MaxPooling2D((2,2),name='third_MaxPooling2D'))
    # ==================================================================================
    
    # ===========================Fourth Block==================================================
    #adding convolution layer
    model.add(Conv2D(128,(3,3),activation=activation,padding="same",name='fourth_Conv2D'))
    #adding pooling layer
    model.add(MaxPooling2D((2,2),name='fourth_MaxPooling2D'))
    # ==================================================================================
    
    #adding fully connected layers
    model.add(Flatten())
    for i in range(n_hidden):
        model.add(Dense(n_neurons,activation=activation,name="fully_connected_layer_"+str(i+1)))
        if add_batch_normalize:
            model.add(BatchNormalization(name="batch_normalizion_"+str(i+1)))
    
    #adding output layer
    model.add(Dense(output_len,activation='softmax',name="output_layer"))
    #compiling the model
    if learning_rate:
        model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer(learning_rate=learning_rate),metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer(),metrics=['accuracy'])
    print(model.summary())
    return model



def fineTuning(X_train,y_train,X_val,y_val):
    # Randomsearchcv    
    keras_clf=KerasClassifier(build_model.CNN_model)
    param_distribs={
        "input_shape":[(128,128,1)],
        "output_len":[10],
        "n_hidden":[1,2,3],
        "n_neurons":[32],
        "learning_rate":[0.002,0.03],
        "optimizer":[tf.keras.optimizers.Adamax],
        "add_batch_normalize":[True,False]
        }

    from sklearn.model_selection import RandomizedSearchCV

    rnd_search=RandomizedSearchCV(keras_clf, param_distribs,n_iter=5,cv=3,verbose=1)  
    # rnd_search.fit(train_norm, trainY, epochs=10,batch_size=256, verbose=1,callbacks=[keras.callbacks.EarlyStopping(patience=150,restore_best_weights=True)])
    rnd_search.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=5)
    rnd_search.best_params_
    rnd_search.best_score_
    model=rnd_search.best_estimator_.model
    return model
    




# # Example
# model=CNN_model((28,28,1),10,n_hidden=1,n_neurons=60,learning_rate=(0.1),optimizer=tf.keras.optimizers.Adamax,activation=tf.keras.layers.LeakyReLU(alpha=0.5),add_batch_normalize=True)
# model.fit(train_norm, trainY, epochs=10,batch_size=256, verbose=1)


# # for plot your model
# plot_model(model, to_file=r'G:\pdf\Level 4\part2\Selected 2\Project\model_plot.png', show_shapes=True, show_layer_names=True)


# # Randomsearchcv

# from keras.wrappers.scikit_learn import KerasClassifier
# keras_clf=KerasClassifier(CNN_model)

# param_distribs={
#     "input_shape":[(28,28,1)],
#     "output_len":[10],
#     "n_hidden":[1,2,3],
#     "n_neurons":[32],
#     "learning_rate":[0.002,0.03],
#     "optimizer":[tf.keras.optimizers.Adamax],
#     "add_batch_normalize":[True,False]
#     }

# from sklearn.model_selection import RandomizedSearchCV

# rnd_search=RandomizedSearchCV(keras_clf, param_distribs,n_iter=5,cv=3,verbose=1)  

# # rnd_search.fit(train_norm, trainY, epochs=10,batch_size=256, verbose=1,callbacks=[keras.callbacks.EarlyStopping(patience=150,restore_best_weights=True)])
# rnd_search.fit(train_norm, trainY, epochs=3,batch_size=256, verbose=1)

# rnd_search.best_params_
# rnd_search.best_score_


# model=rnd_search.best_estimator_.model
# model.fit(train_norm, trainY, epochs=10,batch_size=256, verbose=1)
