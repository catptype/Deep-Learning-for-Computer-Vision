import sys
sys.dont_write_bytecode = True

import random
import tensorflow as tf
from collections import Counter

class DatasetProcessor():
    """
    A utility class for processing and managing datasets.

    Args:
        data_list (list): A list of tuples containing data paths and labels.

    Attributes:
        data_list (list): The input list of data tuples.
        class_list (list): A sorted list of unique labels present in the dataset.
        class_dict (dict): A dictionary mapping class labels to numerical representations.
        class_weight (dict): A dictionary containing class weights for imbalanced datasets.
        train_data (list): The training data after splitting.
        test_data (list): The testing data after splitting.

    Methods:
        create_class_dict(mode="onehot"): Create a dictionary mapping class labels to numerical representations.
        print_raw(): Print the raw dataset statistics.
        train_test_splitter(test_ratio=10, balance=False): Split the dataset into training and testing sets.
    """
    def __init__(self, data_list):
        self.data_list = data_list
        self.class_list = list(sorted(set(label for _, label in self.data_list)))
        self.class_dict = None
        self.class_weight = None
        self.train_data = None
        self.test_data = None

    # Private methods
    def __balance_data(self):
        # Count the occurrences of each class
        class_counts = Counter(label for _, label in self.data_list)
        
        # Ensure that pickup is less than or equal to the minimum class count when balancing
        min_count = min(class_counts.values())

        # Randomly select data points for each class and add them to the output list
        output_data_list = []
        for class_item in self.class_list:
            class_data = [(path, data_class) for path, data_class in self.data_list if data_class == class_item]
            selected_data = random.sample(class_data, min_count)
            output_data_list.extend(selected_data)

        return output_data_list
    
    def __calculate_class_weight(self):
        # Initialize label counter
        label_count = {label:0 for label in self.class_list}

        # Counting label in train data
        for _, label in self.train_data:
            label_count[label] += 1

        # Calculate class weight
        total = sum([number for number in label_count.values()])
        num_class = len(self.class_list)
        self.class_weight = {idx: total / (num_class * label_count[label_key]) for idx, label_key in enumerate(label_count.keys())}
   
    def __print_train_test(self):
        data_dict = {"train": self.train_data, "test": self.test_data}
        
        for train_test, data in data_dict.items():
            print(f"{len(data)} {train_test} data including")
            
            for class_item in self.class_list:
                class_count = sum(1 for _, dump_class in data_dict[train_test] if dump_class == class_item)
                print(f"{class_item}: {class_count}")
            
            print("==========")
    
    def __print_class_weight(self):
        for idx, class_item in enumerate(self.class_list):
            print(f"{class_item} class has weight: {self.class_weight[idx]:.4f}")

    # Public methods
    def create_class_dict(self, mode="onehot"):
        # Convert class to integer number
        int_class = [idx for idx, _ in enumerate(self.class_list)]

        if mode == "onehot":
            # Convert integer number to one-hot vector category
            one_hot_vector = tf.keras.utils.to_categorical(int_class)
            self.class_dict = {label: one_hot for label, one_hot in zip(self.class_list, one_hot_vector)}
        
        elif mode == "index":
            self.class_dict = {label: index for label, index in zip(self.class_list, int_class)}
        else:
            raise ValueError("mode has values \"onehot\" and \"index\"")
    
    def print_raw(self):
        print(f"{len(self.data_list)} data including")
        for class_item in self.class_list:
            class_count = sum([1 for _, dump_class in self.data_list if dump_class == class_item])
            print(f"{class_item}: {class_count}")

    def train_test_splitter(self, test_ratio=10, balance=False):
        if balance:
            data_list = self.__balance_data()
        else:
            data_list = self.data_list

        self.train_data = []
        self.test_data = []
        
        for class_item in self.class_list:
            data_class_list = [(path, data_class) for path, data_class in data_list if data_class == class_item]
            test_size = round(len(data_class_list) * (test_ratio / 100))
            
            # Randomly select test samples
            test_class_list = random.sample(data_class_list, test_size)
            
            # Remove test samples from the data_class_list
            data_class_list = [elem for elem in data_class_list if elem not in test_class_list]
            
            # Add the remaining samples to the training set
            train_class_list = data_class_list
            
            self.train_data.extend(train_class_list)
            self.test_data.extend(test_class_list)

        self.__print_train_test()
        if not balance:
            self.__calculate_class_weight()
            self.__print_class_weight()