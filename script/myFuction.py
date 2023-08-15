import os
import sys
import random
import tensorflow as tf
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

sys.dont_write_bytecode = True

def dir_file_tree(root_path, padding=''):
    dirs = os.listdir(root_path)
    dirs.sort()
    for dir in dirs:
        file_path = os.path.join(root_path, dir)
        if os.path.isdir(file_path):
            subdir_path = os.path.join(root_path, dir)
            sub_files = os.listdir(subdir_path)
            num_files = sum(1 for f in sub_files if os.path.isfile(os.path.join(subdir_path, f)))
            if num_files > 0:
                print(f"{padding}├── {dir} \t ({num_files} files)")
            else:
                print(f"{padding}├── {dir}")
            dir_file_tree(file_path, padding + '│   ')
        else:
            print(f"{padding}├── {dir}")
    pass

def dir_count_tree(root_path, padding=''):
    dirs = os.listdir(root_path)
    dirs.sort()
    for dir in dirs:
        file_path = os.path.join(root_path, dir)
        if os.path.isdir(file_path):
            subdir_path = os.path.join(root_path, dir)
            sub_files = os.listdir(subdir_path)
            num_files = sum(1 for f in sub_files if os.path.isfile(os.path.join(subdir_path, f)))
            if num_files > 0:
                print(f"{padding}├── {dir} \t ({num_files} files)")
            else:
                print(f"{padding}├── {dir}")
            dir_count_tree(file_path, padding + '│   ')
    pass

def pick_data_per_class(data_list, pickup):
    class_list = list({class_list for _, class_list in data_list})
    output_data_list = []

    label_list = [label for _, label in data_list]
    label_count = {}

    for class_item in class_list:
        label_count[class_item] = label_list.count(class_item)

    min_label = min([val for val in label_count.values()])
    
    assert pickup <= min_label, f"pickup:({pickup}) must <= minimum label:({min_label})"

    for class_item in class_list:
        data_class_list = [(path, data_class) for path, data_class in data_list if data_class == class_item]
        temp_list = random.sample(data_class_list, pickup)

        output_data_list += temp_list

    return output_data_list

def balance_data(data_list):
    class_list = list({class_list for _, class_list in data_list})
    balance_data_list = []

    label_list = [label for _, label in data_list]
    label_count = {}

    for class_item in class_list:
        label_count[class_item] = label_list.count(class_item)

    min_label = min([val for val in label_count.values()])
    
    for class_item in class_list:
        data_class_list = [(path, data_class) for path, data_class in data_list if data_class == class_item]
        temp_list = random.sample(data_class_list, min_label)

        balance_data_list += temp_list
    
    return balance_data_list

def split_data(all_list, ratio=10):
    class_list = list({class_list for _, class_list in all_list})
    train = []
    test = []
    for class_item in class_list:
        data_class_list = [(path, data_class) for path, data_class in all_list if data_class == class_item]
        test_class_list = random.sample(data_class_list, round(len(data_class_list)*(ratio/100)))
        for elem in test_class_list:
            data_class_list.remove(elem)
        train_class_list = data_class_list

        train += train_class_list
        test += test_class_list
    
    return train, test

def create_dict_class(data_list):
    """
    data list must be [(data1, label1), (data2, label2)]
    """

    # Export class list from image list
    class_list = list({class_list for _, class_list in data_list})
    class_list = sorted(class_list)

    # Convert class to integer number
    int_class = [idx for idx, _ in enumerate(class_list)]
    
    # Convert integer number to one ot vector category
    one_hot_vector = tf.keras.utils.to_categorical(int_class)

    dict_class = {label:one_hot for label, one_hot in zip(class_list,one_hot_vector)}
    return class_list, dict_class

def calculate_class_weight(data_list, index_mode=True):
    """
    data list must be [(data1, label1), (data2, label2), ...]
    """
    
    # Export class list from image list
    class_list = list({class_list for _, class_list in data_list})
    class_list = sorted(class_list)

    # Initialize label counter
    label_count = {label:0 for label in class_list}

    # Counting label
    for _, label in data_list:
        label_count[label] += 1

    # Calculate class weight
    total = sum([number for number in label_count.values()])
    if index_mode:
        class_weight = {idx: total / (len(class_list) * label_count[label_key]) for idx, label_key in enumerate(label_count.keys())}
    else:
        class_weight = {label_key: total / (len(class_list) * label_count[label_key]) for label_key in label_count.keys()}
    return class_weight

def view_random_image_dataset(dir,random_num=5,rows=1,cols=5,figsize=(16,7)):
    #if condition returns False, AssertionError is raised:
    assert rows*cols >= random_num, f"random_num({random_num}) <= rows*cols({rows*cols})"
    
    # Get all image list
    image_list = [os.path.join(root,image) for root, _, files in os.walk(dir, topdown=True) for image in files]
    # Random pick up for n images
    random_image = random.sample(image_list,random_num)

    plt.figure(figsize=figsize)
    for idx, image_path in enumerate(random_image):
        
        ##########
        # This clode can be edited if the file path is different
        ##########
        info_list  = image_path.split("\\")
        main_class = info_list[1]
        sub_class  = info_list[2]
        ##########

        image = mpimg.imread(image_path)
        plt.subplot(rows, cols, idx+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Class: {main_class}\nSub: {sub_class}\nShape: {image.shape}')
    pass

def view_batch_image(dataset, figsize=(10,10)):
    """
    Randomly view image from batch dataset
    """
    plt.figure(figsize=figsize)
    for images, _ in dataset.take(1):
        
        # Check how many rows-cols
        base = 2
        while True:
            if math.log(len(images), base) <= 2:
                break   
            base += 1

        for i in range(len(images)):
            plt.subplot(base, base, i + 1)
            plt.imshow((images[i]*255).numpy().astype("uint8"))
            plt.axis("off")
    pass

def view_batch_ImageDataGenerator(dataset, class_list, figsize=(10,10)):
    """
    View image from batch dataset which is generated by ImageDataGenerator
    """
    images, labels = next(dataset)
    labels = [class_list[np.argmax(one_hot)] for one_hot in labels]

    num_image = len(images)

    # Check how many rows-cols
    base = 2
    while True:
        if math.log(num_image, base) <= 2:
            break   
        base += 1

    plt.figure(figsize=figsize)
    for i in range(num_image):
        plt.subplot(base, base, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {labels[i]}\nShape: {images[i].shape}")
        plt.axis('off')
    plt.show()
    pass