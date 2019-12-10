from os import listdir, makedirs
from os.path import join, basename, exists

import matplotlib.pyplot as plt
from PIL import Image
import re
import sys
import argparse

def show_image(image_file_path, image_file_name):
    image = plt.imread(image_file_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(image_file_name)
    ax.axis('off')  # clear x-axis and y-axis
    plt.show()

def get_number(filename):
    filename_parts = re.split("_|\.", filename)
    return int(filename_parts[1])

def sort_files(dir_to_organize):
    image_files = []
    for namefile in listdir(dir_to_organize):
        if namefile.endswith(".jpg"):
            image_files.append(namefile)
    image_files.sort(key=get_number)
    return image_files

def save_to_new_dataset(image_file_path, categories_dir, category_chosen):
    image = Image.open(image_file_path)
    image_name = "{}_{}.jpg".format(category_chosen, get_number(basename(image_file_path)))
    new_image_file_path = join(categories_dir, join(category_chosen, image_name))
    image.save(new_image_file_path)

def organize_files(dir_to_organize, categories_dir, NEW_CATEGORIES):
    image_files = sort_files(dir_to_organize)
    message_classify = ""
    for index, category in NEW_CATEGORIES.items():
        message_classify += "Type {} to choose {} and -1 to exit\n".format(index, category)
    
    first_number = int(get_number(image_files[0]))
    last_number = int(get_number(image_files[-1]))
    message_index = "Type an index to start from {} to {}: ".format(first_number, last_number)
    
    answer = input(message_index)
    while True:
        try:
            from_index = int(answer)
            if from_index < first_number or from_index > last_number:
                raise Exception("Index out of range")
            break
        except Exception as e:
            print(e)
    from_index = from_index - first_number
    for i in range(from_index, len(image_files)):
        image_file = image_files[i]
        image_file_path = join(dir_to_organize, image_file)
        print(message_classify)
        show_image(image_file_path, image_file)
        while True:
            index_chosen = input()
            if index_chosen == "-1":
               print("last index:", first_number + i)
               sys.exit(0)
            if index_chosen in NEW_CATEGORIES.keys():
                break
            print("Error in the input, remeber:", message_classify)
        
        save_to_new_dataset(image_file_path, categories_dir, NEW_CATEGORIES[index_chosen])
    print("Finished")

if __name__== "__main__":
    while True:
        try:
            answer = input("Type 0 to organize train images or 1 to organize test images: ")
            if answer == "0":
                organize_train = True
            elif answer == "1":
                organize_train = False
            else:
               raise Exception("Error in the input, try again")
            
            answer = input("Type 0 to organize organic images or 1 to organize recyclable images: ")
            if answer == "0":
                organize_organic = True
            elif answer == "1":
                organize_organic = False
            else:
               raise Exception("Error in the input, try again")
            break
        except Exception as e:
            print(e)
    
    print()
    DATASET_DIR = "waste-classification-data/DATASET/"
    NEW_DATASET_DIR = "NEW_DATASET"
    NEW_CATEGORIES = {"0" :"Recyclable", "1": "Ordinary", "2": "Organic"}

    if organize_train:
        dir_to_organize = join(DATASET_DIR, "TRAIN")
        categories_dir = join(NEW_DATASET_DIR, "TRAIN")
    else:
        dir_to_organize = join(DATASET_DIR, "TEST")
        categories_dir = join(NEW_DATASET_DIR, "TEST")
            
    if organize_organic:
        dir_to_organize =  join(dir_to_organize, "O")
    else:
        dir_to_organize =  join(dir_to_organize, "R")

    if not exists(dir_to_organize):
        raise Exception("The path {} doesn't exist".format(dir_to_organize))
        
    for category in NEW_CATEGORIES.values():
        category_dir = join(categories_dir, category)
        if not exists(category_dir):
            makedirs(category_dir)
    organize_files(dir_to_organize, categories_dir, NEW_CATEGORIES)
