import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .DirectoryProcessor import DirectoryProcessor

class ImageOrganizer:
    """
    Utility class for organizing images into labeled directories based on a classification model.

    Parameters:
        h5_file (str): Path to the pre-trained Keras model file in HDF5 format.
        target_dir (str): Directory containing images to be classified and organized.
        batch_size (int): Batch size for image processing (default is 8).
        threshold_score (float): Threshold score for classifying images (default is 0.9).

    Attributes:
        model (tf.keras.Model): Loaded Keras model for image classification.
        target_dir (str): Directory containing images to be classified and organized.
        batch_size (int): Batch size for image processing.
        threshold_score (float): Threshold score for classifying images.

    Public Methods:
        image_classify(label_list=None): Classifies and organizes images based on the pre-trained model.

    Private Methods:
        __image_resize(image_path): Reads and resizes an image to the required input shape for the classification model.
        __predict(image_path): Performs class prediction on a batch of images and post-processes the results.

    Example:
        ```python
        # Example usage of ImageOrganizer class
        organizer = ImageOrganizer(h5_file="model.h5", target_dir="images_to_classify", batch_size=16, threshold_score=0.8)
        organizer.image_classify(label_list=["cat", "dog", "bird"])
        ```
    """
    def __init__(self, h5_file, target_dir, batch_size=8, threshold_score=0.9):
        print("Loading model ... ", end="")
        self.model = load_model(h5_file)
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.threshold_score = threshold_score
        print("COMPLETE")

    # Private methods
    def __image_resize(self, image_path):
        _, height, width, _ = self.model.input_shape
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(
            image, 
            (height, width), 
            method = "bilinear", 
            preserve_aspect_ratio = True, 
            antialias = True,
        )
        image = tf.image.resize_with_pad(image, height, width)
        image = image / 255.0
        
        return image
    
    def __predict(self, image_path):
        print("Processing class prediction ...")

        prediction_result = []

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path)
        image_dataset = image_dataset.map(self.__image_resize)
        image_batch = image_dataset.batch(self.batch_size)

        progress_bar = tf.keras.utils.Progbar(
            len(image_batch), 
            width=20, 
            interval=0.2, 
            unit_name='batch',
        )

        for batch in image_batch:
            prediction_batch = self.model.predict(batch)
            class_score_batch = np.max(prediction_batch, axis=-1)  # Get class scores
            class_idx_batch = np.argmax(prediction_batch, axis=-1)

            # Post processing thresholding score
            for idx, class_score in enumerate(class_score_batch):
                if class_score < self.threshold_score:
                    class_idx_batch[idx] = -1
            
            prediction_result.extend(class_idx_batch)
            progress_bar.add(1)
        
        # Release memory after finish prediction
        tf.keras.backend.clear_session()

        return prediction_result

    # Public method
    def image_classify(self, label_list=None):
        if label_list is not None:
            label_list.append("UNKNOWN")

        image_path_list = DirectoryProcessor.get_only_files(self.target_dir, [".jpg", ".png"], include_sub_dir=False)
        predict_result = self.__predict(image_path_list)
        
        print("Moving files ...")
        progress_bar = tf.keras.utils.Progbar(
            len(image_path_list),
            width = 20, 
            interval = 0.2, 
            unit_name = 'image',
        )
        
        for image_path, predict in zip(image_path_list, predict_result):
            directory, filename = os.path.split(image_path)
            sub_directory = f"class_{predict}" if label_list is None else f"class_{label_list[predict]}"

            source = os.path.join(directory, filename)
            destination = os.path.join(directory, sub_directory, filename)

            DirectoryProcessor.move_file(source, destination)
            progress_bar.add(1)