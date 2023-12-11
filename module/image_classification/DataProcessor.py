import random
import tensorflow as tf
from collections import Counter

class DataProcessor:
    """
    Utility class for processing dataset.

    Methods:
        balance_data(data_list):
            Balances the provided data list by randomly selecting the minimum occurrences
            of each class and creating a new balanced data list.

        create_label_dict(label_list, mode):
            Creates a dictionary mapping labels to either one-hot vectors or integer indices.

        validation_spliter(data_list, label_set, ratio):
            Splits the data into training and testing sets based on the given ratio for each class.

        calculate_class_weight(train_dataset):
            Calculates class weights for imbalanced datasets.

    Note: All methods are static and do not require an instance of the class.
    """

    @staticmethod
    def balance_data(data_list):
        # Count the occurrences of each class and find the minimum number
        label_list = [label for _, label in data_list]
        label_set = sorted(set(label_list))
        label_counts = Counter(label for label in label_list)
        min_count = min(label_counts.values())

        # Randomly select data points for each class and add them to the output list
        balanced_data_list = []
        for label_current in list(label_set):
            data_label_list = [(data, label) for data, label in data_list if label == label_current]
            selected_data = random.sample(data_label_list, min_count)
            balanced_data_list.extend(selected_data)

        return balanced_data_list

    @staticmethod
    def create_label_dict(label_list, mode):
        if mode not in ["onehot", "index"]:
            raise ValueError("mode must be 'onehot' or 'index'")
        
        if mode == "onehot":
            label_index = [idx for idx, _ in enumerate(label_list)]
            one_hot_vector = tf.keras.utils.to_categorical(label_index)
            label_dict = {label: one_hot for label, one_hot in zip(label_list, one_hot_vector)}
        elif mode == "index":
            label_dict = {label: index for index, label in enumerate(label_list)}
        
        return label_dict
    
    @staticmethod
    def validation_spliter(data_list, label_set, ratio):
        train_data, test_data = [], []
        
        for label_current in label_set:
            data_label_list = [(data, label) for data, label in data_list if label == label_current]
            test_size = round(len(data_label_list) * ratio)
            
            # Randomly select test samples
            test_class_list = random.sample(data_label_list, test_size)
            test_data.extend(test_class_list)
            
            # Remove test samples from the data_label_list
            data_label_list = [elem for elem in data_label_list if elem not in test_class_list]
            train_data.extend(data_label_list)
            
        return train_data, test_data
    
    @staticmethod
    def calculate_class_weight(train_dataset):
        label_count = Counter(label for _, label in train_dataset)
        total = sum(label_count.values())
        num_class = len(label_count)
        class_weight = {label: round(total / (num_class * count), 4) for label, count in label_count.items()}
        return class_weight