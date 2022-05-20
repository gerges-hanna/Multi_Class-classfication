import os
import cv2
import logging
from math import ceil
from threading import Thread
import shutil
import imgaug as ia
import time
from datetime import timedelta

ia.seed(4)

from augmentation_functions import (
    zoom, rotation, flip,
    add_noise_using_imgaug,
    blur, crop_and_rotate
)

def split_list(objects: list, pieces: int) -> list:
    """
    This function splits list to multiple list
    every sublist contain number of objects
    from the original list, the number of sublists cannot 
    exceed the number of pieces
    """
    listLen = len(objects)
    start = 0
    end = ceil(listLen / pieces)
    output = []
    while any(objects):
        output.append(objects[start:end])
        del objects[start:end]
    return output

def augment_data_single_thread(fruits: list, old_data_path: str, new_data_path: str, goal=300):
    for fruit_name in fruits:
        new_directory_absolute_path = os.path.join(new_data_path, fruit_name)
        make_new_directory(new_directory_absolute_path)
        augmentation_functions = {
            1: {"function": lambda x, *args: x, "parameters": [None], "name":"_original"},
            2: {"function": flip , "parameters":  [1], "name":"_flipped_H"},
            3: {"function": flip , "parameters": [0], "name":"_flipped_V"},
            4: {"function": blur , "parameters": [20], "name":"_blurred"},
            5: {"function": add_noise_using_imgaug , "parameters": [(10, 60)], "name":"_noised"},
            6: {"function": crop_and_rotate, "parameters": [0.25, 300], "name":"_cropped_rotated"},
        }
        def applying_augmentation(augment_function_index=1):
            if not augmentation_functions.get(augment_function_index):
                return
            if augment_function_index == 1:
                fruits_images = os.listdir(os.path.join(old_data_path, fruit_name))
            else:
                fruits_images = os.listdir(os.path.join(new_data_path, fruit_name))
            remained_augmentation_goal = goal - len(fruits_images)
            if remained_augmentation_goal <= 0:
                return
            for index, image in enumerate(fruits_images):
                if index == remained_augmentation_goal:
                    return
                image_absolute_path_new_data = os.path.join(new_data_path, fruit_name, image)
                if augment_function_index == 1:
                    image_absolute_path_original_data = os.path.join(old_data_path, fruit_name, image)
                else:
                    image_absolute_path_original_data = os.path.join(new_data_path, fruit_name, image)
                original_image = cv2.imread(image_absolute_path_original_data, cv2.IMREAD_UNCHANGED)
                augmentation_details = augmentation_functions.get(augment_function_index)
                augmented_image = augmentation_details.get("function")(original_image, *augmentation_details.get("parameters"))
                splitted_image_name = image_absolute_path_new_data.split(".")
                splitted_image_name.insert(1, augmentation_details.get("name")+".")
                augmented_image_file_name = "".join(splitted_image_name)
                cv2.imwrite(augmented_image_file_name , augmented_image)
            fruits_images.clear()
            
            applying_augmentation(augment_function_index+1)
        applying_augmentation()

def make_new_directory(directory_absolute_path):
    try:
        os.mkdir(directory_absolute_path)
    except FileExistsError:
        logging.warning(f"{directory_absolute_path} already exists")


   
if __name__ == "__main__":
    
    data_parent_directory_path = input("Enter absolute path of dataset: \n")
    new_data_parent_directory_path = input("Enter Output Folder: \n")        
    directories_of_fruits = os.listdir(data_parent_directory_path)
    cores = os.cpu_count()
    
    print("Start Spliting Data into classes...")
    splitted_fruits = split_list(directories_of_fruits, cores)
    print("Data Spliting Done.")

    threads = []
    start_time = time.monotonic()
    
    print("Start initilizing and start threads...")
    
    # creating and starting threads according to the number of sublists in splittedList
    for thread_counter in range(len(splitted_fruits)):
        thread = Thread(target=augment_data_single_thread, kwargs={'fruits': splitted_fruits[thread_counter], 'old_data_path': data_parent_directory_path, 'new_data_path': new_data_parent_directory_path})
        # thread = Thread(target=check_images, kwargs={'fruits': splitted_fruits[thread_counter], 'old_data_path': data_parent_directory_path})
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()    
    print("Data has been augmented.")
    
    
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

