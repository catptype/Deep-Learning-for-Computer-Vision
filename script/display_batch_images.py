import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from matplotlib import pyplot as plt

def display_batch_images(dataset, class_list, figsize=(10, 10)):
    """
    Display a batch of images and their associated classes.

    Args:
        dataset (tf.data.Dataset): The dataset containing image batches.
        class_list (list): A list of class labels corresponding to the dataset.
        figsize (tuple): The figure size (width, height) in inches (default is (10, 10)).

    Example:
        To display a batch of images and their associated classes from a TensorFlow dataset `train_dataset` 
        with class labels specified in `class_list`, you can use the following code:

        ```python
        class_list = ['class1', 'class2', 'class3']  # Replace with your class labels
        display_batch_images(train_dataset, class_list, figsize=(12, 12))
        ```

    Note:
        This function assumes that the labels in the dataset are either one-hot encoded or index label numbers.
        If the labels are one-hot encoded, they will be converted to string labels using `class_list`.
        If the labels are index label numbers, they will also be converted to string labels using `class_list`.
    """
    plt.figure(figsize=figsize)
    for images, labels in dataset.take(1):
        base = 2
        while True:
            if math.log(len(images), base) <= 2:
                break   
            base += 1

        if labels.shape[1] == len(class_list):
            # Labels are one-hot encoded, convert to string labels
            labels = [class_list[np.argmax(one_hot)] for one_hot in labels]
        else:
            # Labels are index label numbers, convert to string labels
            labels = [class_list[label] for label in labels]

        for i in range(len(images)):
            plt.subplot(base, base, i + 1)
            plt.imshow((images[i] * 255).numpy().astype("uint8"))
            plt.title(f"Class: {labels[i]}\nShape: {images[i].shape}")
            plt.axis("off")
