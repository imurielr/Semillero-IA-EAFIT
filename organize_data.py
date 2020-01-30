from os import listdir, makedirs
from os.path import join, basename, exists

import matplotlib.pyplot as plt
from PIL import Image
import re
import sys


def show_image(image_file_path, image_file_name):
    """
    Show a image in a new window. 
    Arguments:
        image_file_path  -- the full image path
        image_file_name  -- the name of the image
    """
    image = plt.imread(image_file_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(image_file_name)
    ax.axis('off')  # clear x-axis and y-axis
    plt.show()

def get_number(filename):
    """Get the number from a photo name, for example we want to get 12568 from O_12568.jpg
    
    Arguments:
        filename -- A photo name including the extension.
    Returns:
        int -- The number from the photo name
    """
    filename_parts = re.split("_|\.", filename) # For example the image O_12568.jpg to get [O, 12568, jpg]
    return int(filename_parts[1])

def get_image_filenames(dir_to_organize):
    """
    Get the list with all the image filenames from the directory to organize
    Arguments:
        dir_to_organize -- The path to the directory that will be organized
    
    Returns:
        list --A with all the image filenames from the directory to organize
    """
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
    """
    Execute the program to let to the user organize the data.
    Arguments:
        dir_to_organize  -- The path to the directory that will be organized
        categories_dir  --  The path to save the photos with the new label
        NEW_CATEGORIES  --  The dictionary with all the names of the new categories.
    
    Raises:
        Exception: If you type an incorrect index
    """
    image_files = get_image_filenames(dir_to_organize)
    
    first_number = int(get_number(image_files[0])) # The first number of the photos in the directory to organize
    last_number = int(get_number(image_files[-1])) # The last number of the photos in the directory to organize
    message_index = "Type an index to start, it has to be from {} to {}: ".format(first_number, last_number)
    
    # get an index to start
    while True:
        try:
            answer = input(message_index)
            from_index = int(answer)
            if from_index < first_number or from_index > last_number:
                raise Exception("Index out of range")
            break
        except Exception as e:
            print(e)
    
    message_classify = ""
    for index, category in NEW_CATEGORIES.items():
        message_classify += "Type {} to choose {} and -1 to exit\n".format(index, category)

    from_index -= first_number # To get the list position of the "index to start"
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
            print("Error in the input, remember:", message_classify)
        save_to_new_dataset(image_file_path, categories_dir, NEW_CATEGORIES[index_chosen])
    
    print("Finished")

def usage():
    print("You have to include the path of the dataset")
    exit(1)

if __name__== "__main__":
    if len(sys.argv) <= 1:
        usage()
   
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

    DATASET_DIR = sys.argv[1]
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
    
    # create dirs for the new dataset    
    for category in NEW_CATEGORIES.values():
        category_dir = join(categories_dir, category)
        if not exists(category_dir):
            makedirs(category_dir)
    organize_files(dir_to_organize, categories_dir, NEW_CATEGORIES)
