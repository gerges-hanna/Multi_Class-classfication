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




def preprocessing(img_path, size=128, grayScale=True):
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



            
            


            
            
import matplotlib.pyplot as plt          
from sklearn.metrics import confusion_matrix          
    
#Defining function for confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, classes,  cmap=plt.cm.Blues):

    
    #Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)   
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    #Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    return ax