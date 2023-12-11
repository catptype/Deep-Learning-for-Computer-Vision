import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from .DirectoryProcessor import DirectoryProcessor

class ImageFeatureExtractor:
    """
    Utility class for extracting feature vectors from images using a pre-trained model.

    Parameters:
        h5_file (str): Path to the HDF5 file containing the pre-trained model.
        batch_size (int): Batch size for processing images in batches.

    Attributes:
        model (tf.keras.Model): The loaded pre-trained model for feature extraction.
        batch_size (int): Batch size for processing images in batches.

    Public Methods:
        __init__(h5_file, batch_size=64): Initializes the ImageFeatureExtractor object.
        export_json(target_dir): Exports the calculated feature vectors to a JSON file.

    Private Methods:
        __image_resize(image_path): Resizes an image to match the input size of the pre-trained model.
        __padding(image): Pads an image to the input size of the pre-trained model.
        __preprocessing(image_path): Applies image resizing and padding for preprocessing.
        __calculate_feature(image_path): Calculates feature vectors for a list of images.

    Example:
        ```python
        # Example usage of ImageFeatureExtractor class
        feature_extractor = ImageFeatureExtractor('model.h5', batch_size=32)
        feature_extractor.export_json(['data/dataset1', 'data/dataset2'])
        ```

    """
    def __init__(self, h5_file, batch_size=64):
        print("Load model ... ", end="")
        self.model = load_model(h5_file, compile=False)
        self.batch_size = batch_size
        print("COMPLETE")
    
    # Private methods
    @tf.autograph.experimental.do_not_convert
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
        image = image / 255.0
        return image
    
    @tf.autograph.experimental.do_not_convert
    def __padding(self, image):
        image_height, image_width = image.shape[:2] # OpenCV format
        _, target_width, target_height, _ = self.model.input_shape

        # Calculate the padding sizes for both width and height
        pad_width = max(0, target_width - image_width)
        pad_height = max(0, target_height - image_height)

        # Calculate the padding amounts for top, bottom, left, and right
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Padding image
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REPLICATE)
        
        image = tf.convert_to_tensor(image)
        return image

    @tf.autograph.experimental.do_not_convert
    def __preprocessing(self, image_path):
        image = self.__image_resize(image_path)
        image = tf.numpy_function(
            func = self.__padding, 
            inp = [image], 
            Tout = tf.float32, 
            name="Image_padding",
        )
        image = tf.ensure_shape(image, (self.model.input_shape[1], self.model.input_shape[2], self.model.input_shape[3]))
        return image
    
    def __calculate_feature(self, image_path):
        feature_list = []

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path)
        image_dataset = image_dataset.map(self.__preprocessing)
        image_batch = image_dataset.batch(self.batch_size)

        progress_bar = tf.keras.utils.Progbar(
            len(image_batch), 
            width=20, 
            interval=0.1, 
            unit_name='batch',
        )

        for batch in image_batch:
            feature_list.extend(self.model.predict(batch))
            progress_bar.add(1)
        
        # Release memory after finish prediction
        tf.keras.backend.clear_session()
        return feature_list
    
    # Public method
    def export_json(self, target_dir):
        json_path = f"{self.model.name}.json"
        image_path_list = []

        for dir in target_dir:
            image_path_list.extend(DirectoryProcessor.get_only_files(dir, [".jpg", ".png"], include_sub_dir=True))
        
        print("Calculate database feature vector ...")
        vector_list = self.__calculate_feature(image_path_list)

        print("Export JSON file ... ", end="")
        json_data = [{"path": image_path, "feature_vector": vector.tolist()} for image_path, vector in zip(image_path_list, vector_list)]

        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)
        print("COMPLETE")