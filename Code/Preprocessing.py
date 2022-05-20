# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:29:58 2022

@author: BEB0
"""
import os
import glob 
from skimage import io 
import skimage
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def get_classes(dataset_path):
    class_names = os.listdir(dataset_path)
    print(class_names)
    return class_names
   
def read_dataset(dataset_path):
    data = []
    labels = []
    classes_name=get_classes(dataset_path)
    for img_class in classes_name:
        #get name of all images path within img_class 
        images_paths = os.path.join( dataset_path, img_class ,'*')
        images = glob.glob(images_paths)
        lbl=[img_class]*len(images)
        labels=labels+lbl
        data=data+images
    return data, labels

def preprocessing(img_path, size=128, grayScale=False):
    """
    # fixed size parameter 
    # normalization
    # gray prarmeter 
    # scal small imges 
    """
    image = io.imread(img_path)
    #resize images 
    image = cv2.resize(image, (size, size)) 
    #images normalization 
    image=image/255
    #Grayscaleing images from colored to black and white
    if grayScale:
        image = skimage.color.rgb2gray(image)    

    return image



            
            
def split_train_test_validation(X,y,test_size,val_size):
    remain=(test_size+val_size)
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=remain,random_state=42,stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=(val_size/remain),random_state=42,stratify=y_remain)
    return X_train,X_test,X_val,y_train,y_test,y_val

            
  