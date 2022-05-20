
import  os
from  Preprocessing import preprocessing
from  Preprocessing import get_classes
from  Preprocessing import read_dataset
from  Preprocessing import split_train_test_validation
import build_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.utils.vis_utils import plot_model
import keras
import matplotlib.pyplot as plt
 

    

if __name__ == "__main__":
    # Importing and Loading the data into a data frame
    dataset_path = input("Enter Dataset Path: \n")
    # dataset_path = r"C:\Users\Gerges_Hanna\Downloads\Selected2_Project\Delete\Data\Dataset"
    os.chdir(dataset_path)
    print("Reading Dataset...")
    # get classes of classifications 
    classes_name = get_classes(dataset_path)
    print("Classes Name:", classes_name)
    
    #read dataset
    data, labels = read_dataset(dataset_path)
    print("Dataset has been readed.")
    print("Start Preprocessing...")
    #preprocess data
    convert2gray=False
    data=[preprocessing(item,size=128,grayScale=(convert2gray)) for item in data]
    # store data in numpy array
    x = np.array(data)
    #convert string labales to one hot encoding 
    one_hot = LabelEncoder()
    y = one_hot.fit_transform(labels)
    #split data into train and test 
    X_train,X_test,X_val,y_train,y_test,y_val = split_train_test_validation( x, y, 0.2, 0.2)
    print("Preprocessing finished.")
    print("Start Buliding Model...")
    #build the model 
    if convert2gray:
        dim=1
    else:
        dim=3
    model = build_model.CNN_model((128,128,dim),7,n_hidden=1,n_neurons=32, activation = "relu", add_batch_normalize=(True) )
    model.optimizer
    #fitting the model and get summary 
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=10, callbacks=[keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
    print("Model has been bulid.")
    print("Model Summary:")    
    model.summary()
    print("Train Loss & Acc:",model.evaluate(X_train,y_train))
    print("Val Loss & Acc:",model.evaluate(X_val,y_val))
    print("Test Loss & Acc:",model.evaluate(X_test,y_test))
