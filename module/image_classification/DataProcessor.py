import random
import tensorflow as tf
from collections import Counter

class DataProcessor:
    """
    A class for data processing tasks, such as balancing data, creating label dictionaries,
    splitting data into training and testing sets, and calculating class weights.

    Methods:
        balance_data(data_list): Balance the provided data by randomly selecting the same number of samples for each class.
        create_label_dict(label_list, mode): Create a label dictionary mapping class labels to one-hot vectors or indices.
        validation_spliter(data_list, label_set, ratio): Split data into training and testing sets according to the specified ratio.
        calculate_class_weight(train_dataset): Calculate class weights for use in imbalanced datasets.
    """
    @staticmethod
    def balance_data(data_list):
        """
        Balance the provided data by randomly selecting the same number of samples for each class.

        Parameters:
            data_list (list): A list of data samples where each sample is a tuple (data, label).

        Returns:
            list: A balanced list of data samples.
        """
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
        """
        Create a label dictionary mapping class labels to one-hot vectors or indices.

        Parameters:
            label_list (list): A list of unique class labels.
            mode (str): The mode for creating label dictionaries, either 'onehot' or 'index'.

        Returns:
            dict: A dictionary mapping class labels to one-hot vectors or indices.
        """
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
        """
        Split data into training and testing sets according to the specified ratio.

        Parameters:
            data_list (list): A list of data samples where each sample is a tuple (data, label).
            label_set (list): A list of unique class labels.
            ratio (float): The ratio of samples to be used for testing (e.g., 0.2 for 20% testing).

        Returns:
            tuple: A tuple containing the training data list and the testing data list.
        """
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
        """
        Calculate class weights for use in imbalanced datasets.

        Parameters:
            train_dataset (list): A list of training data samples where each sample is a tuple (data, label).

        Returns:
            dict: A dictionary mapping class labels to their calculated class weights.
        """
        label_count = Counter(label for _, label in train_dataset)
        total = sum(label_count.values())
        num_class = len(label_count)
        class_weight = {label: round(total / (num_class * count), 4) for label, count in label_count.items()}
        return class_weight
