import  os
os.chdir(r"G:\pdf\Level 4\part2\Selected 2\Project\Project_full")
from  Preprocessing import preprocessing
from  Preprocessing import get_classes
from  Preprocessing import read_dataset
import build_model
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.utils.vis_utils import plot_model

def split_train_test_validation(X,y,test_size,val_size):
    remain=(test_size+val_size)
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=remain,random_state=42,stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=(val_size/remain),random_state=42,stratify=y_remain)
    return X_train,X_test,X_val,y_train,y_test,y_val

# Importing and Loading the data into a data frame
dataset_path = r"G:\pdf\Level 4\part2\Selected 2\Project\Dataset"

# get classes of classifications 
classes_name = get_classes(dataset_path)

#read dataset
data, labels = read_dataset(dataset_path)

#preprocess data
data=[preprocessing(item,size=128,grayScale=(False)) for item in data]

x = np.array(data)

one_hot = LabelEncoder()
y = one_hot.fit_transform(labels)



#split data into train and test 
X_train,X_test,X_val,y_train,y_test,y_val = split_train_test_validation( x, y, 0.2, 0.2)

import keras
#build the model 
model = build_model.CNN_model((128,128,3),7,n_hidden=1,learning_rate=0.001,activation="relu",n_neurons=32,add_batch_normalize=True)
#fitting the model and get summary 
history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=100,callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])

model.summary()




# for plot your model
plot_model(model, to_file=r'G:\pdf\Level 4\part2\Selected 2\Project\model_plot.png', show_shapes=True, show_layer_names=True)


# Randomsearchcv

from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import keras

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
model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=100,callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])



model.evaluate(X_train,y_train)
model.evaluate(X_val,y_val)
model.evaluate(X_test,y_test)
