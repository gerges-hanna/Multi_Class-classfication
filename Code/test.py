# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:22:14 2022

@author: BEB0
"""
import os 
import cv2 as cv
import keras
from Preprocessing import preprocessing


def loadModel(path):
    model = keras.models.load_model(path)
    return model
           
    
def Main():
    
    # load the test image 
    img_path = input("enter the path of the image: ")
    
    #Load the model 
    model_path = input("enter the path of the model: ")
    model = loadModel(model_path)
    
    # preprocessing 
    img = preprocessing(img_path)
    result  = model.predect(img)
    print("The predection: ", result )
    
    
if __name__ == "__main__":
    Main()