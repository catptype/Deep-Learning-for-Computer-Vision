import math
import matplotlib.pyplot as plt
import numpy as np

class ImageVisualizer:
    """
    Utility class for visualizing image datasets.

    Methods:
        display_batch(dataset, label_list, figsize=(10, 10)): Displays a batch of images from the given dataset with corresponding labels.

    Example:
        ```python
        # Example usage of image visualizer
        ImageVisualizer.display_batch(dataset=train_dataset, label_list=class_labels, figsize=(12, 12))
        ```

    Note: This class provides a simple method for visualizing image batches with corresponding labels.
    """
    @staticmethod
    def display_batch(dataset, label_list, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        
        for images, labels in dataset.take(1):
            base = 2
            while True:
                if math.log(len(images), base) <= 2:
                    break   
                base += 1

            if labels.shape[1] == len(label_list):
                # Labels are one-hot encoded, convert to string labels
                labels = [label_list[np.argmax(one_hot)] for one_hot in labels]
            else:
                # Labels are index label numbers, convert to string labels
                labels = [label_list[label] for label in labels]

            for i in range(len(images)):
                image = images[i]
                image = np.clip(image, 0, 1)
                image = (image * 255).astype("uint8")

                plt.subplot(base, base, i + 1)
                plt.imshow(image)
                plt.title(f"Class: {labels[i]}\nShape: {image.shape}")
                plt.axis("off")