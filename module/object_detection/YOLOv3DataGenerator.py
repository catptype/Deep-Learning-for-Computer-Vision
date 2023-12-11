import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

from .YOLOv3Augmentation import YOLOv3Augmentation
from .YOLOv3Anchor import YOLOv3Anchor as anchor_module

class ErrorHandler:
    """
    Utility class for handling errors and input validation in YOLOv3DataGenerator.

    Methods:
        validate_input(input): Validates the input format for YOLOv3DataGenerator.
        validate_image_size(image_size): Validates the image_size parameter format for YOLOv3DataGenerator.
        validate_translation(translate_range): Validates the translate_range parameter format for YOLOv3DataGenerator.
    """
    @staticmethod
    def validate_input(input):
        if not isinstance(input, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in input):
            raise ValueError("Invalid input format")

    @staticmethod
    def validate_image_size(image_size):
        is_valid_tuple = isinstance(image_size, tuple) and len(image_size) == 2 and all(isinstance(dim, int) for dim in image_size)
        is_valid_int = isinstance(image_size, int)
        if not (is_valid_int or is_valid_tuple):
            raise ValueError("Invalid image_size. It should be an integer tuple (width, height) or an integer number")

    @staticmethod
    def validate_translation(translate_range):
        is_valid_tuple = isinstance(translate_range, tuple) and len(translate_range) == 2
        is_valid_float = isinstance(translate_range, float)
        if not (is_valid_tuple or is_valid_float or translate_range is None):
            raise ValueError("Invalid translation format. It should be a tuple (x, y) or a float.")
        
class YOLOv3DataGenerator:
    """
    Data generator for YOLOv3 model training.

    Parameters:
        input (list): List of tuples containing image paths and corresponding XML paths.
        label_list (list): List of class labels.
        image_size (int or tuple): Target size of the augmented image (height, width).
        num_anchor (int): Number of anchor boxes.
        horizontal_flip (bool): Flag for horizontal flip augmentation.
        vertical_flip (bool): Flag for vertical flip augmentation.
        translate_range (float or tuple): Range for random translation.
        rotation_range (float): Maximum rotation angle for random rotation.

    Attributes:
        __input (list): List of tuples containing image paths and corresponding XML paths.
        __label_list (list): List of class labels.
        __image_size (tuple): Target size of the augmented image (height, width).
        __num_anchor (int): Number of anchor boxes.
        __horizontal_flip (bool): Flag for horizontal flip augmentation.
        __vertical_flip (bool): Flag for vertical flip augmentation.
        __translate_range (float or tuple): Range for random translation.
        __rotation_range (float): Maximum rotation angle for random rotation.
        __anchor_boxes (numpy.ndarray): Calculated anchor boxes based on input annotations.

    Public Methods:
        generate_dataset(batch_size, drop_reminder=False): Generates a TensorFlow dataset for YOLOv3 model training.

    Private Methods:
        __image_reader_tf(image_path): Reads and preprocesses an image using TensorFlow.
        __xml_reader(xml_path): Reads and parses XML annotations for an image.
        __generate_annotation_label(annotation_list): Generates annotation labels for YOLOv3 based on the provided annotations.
        __preprocessing(image_path, xml_path): Preprocesses an image and its corresponding XML annotations for YOLOv3 training.

    Example:
        ```python
        # Example usage of YOLOv3DataGenerator class
        data_generator = YOLOv3DataGenerator(input=input_list, label_list=class_labels, image_size=416, num_anchor=9)
        train_dataset = data_generator.generate_dataset(batch_size=32)
        ```
    """
    def __init__(
        self, 
        input,
        label_list,
        image_size, 
        num_anchor, 
        horizontal_flip=False, 
        vertical_flip=False, 
        translate_range=None, 
        rotation_range=None,
    ):  
        # Error handler
        ErrorHandler.validate_input(input)
        ErrorHandler.validate_image_size(image_size)
        ErrorHandler.validate_translation(translate_range)

        # Private artibute    
        self.__input = input
        self.__label_list = label_list
        self.__image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.__num_anchor = num_anchor
        self.__horizontal_flip = horizontal_flip
        self.__vertical_flip = vertical_flip
        self.__translate_range = translate_range
        self.__rotation_range = rotation_range
        
    # Private methods
    def __image_reader_tf(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(
            images = image, 
            size = self.__image_size,
            method = "bilinear", 
            preserve_aspect_ratio = True, 
            antialias = True,
        )
        image = image / 255.0
        return image
    
    def __xml_reader(self, xml_path):
        annotation_list = []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            # Class name or label
            class_name = obj.find('classname').text
            class_idx = self.__label_list.index(class_name)
            
            # Boundary box positions
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            annotation = (class_idx, xmin, ymin, xmax, ymax)
            annotation_list.append(annotation)
        
        annotation_list = tf.convert_to_tensor(annotation_list)
        return annotation_list
    
    def __generate_annotation_label(self, annotation_list):

        label_data = []

        for downsampling_scale in [32, 16, 8]: # YOLOv3 always produce 3 scales prediction
            # Calculate grid size for the current scale
            
            col_grid_size = self.__image_size[0] // downsampling_scale # width
            row_grid_size = self.__image_size[1] // downsampling_scale # height

            # Initialize label data for the current scale
            label_data_shape = (row_grid_size, col_grid_size, self.__num_anchor, 5 + len(self.__label_list))
            scale_label_data = np.zeros(label_data_shape, dtype=np.float32)
            
            for annotation in annotation_list:
                for class_id, xmin, ymin, xmax, ymax in [annotation]:
                
                    # Calculate object center, width, and height
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin

                    # Find the best anchor for the current object based on its size
                    best_anchor = anchor_module.find_best_anchor(width, height, self.__anchor_boxes)

                    # Convert box coordinates and size to grid cell coordinates
                    col_grid_idx = int(x_center * col_grid_size) 
                    row_grid_idx = int(y_center * row_grid_size)

                    # Encode object information into the label_data tensor for the current scale
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 0] = 1.  # Objectness score
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 1] = (x_center * col_grid_size) % 1
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 2] = (y_center * row_grid_size) % 1
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 3] = width
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 4] = height
                    scale_label_data[row_grid_idx, col_grid_idx, best_anchor, 5 + int(class_id)] = 1.  # Class one-hot encoding
            
            label_data.append(scale_label_data)

        return label_data[0], label_data[1], label_data[2]
    
    @tf.autograph.experimental.do_not_convert
    def __preprocessing(self, image_path, xml_path):

        print("CALLING preprocessing function")
        print("READ IMAGE ... ", end="")
        image = self.__image_reader_tf(image_path)
        print("COMPLETE")
        
        print("READ XML ... ", end="")
        annotation_list = tf.numpy_function(
            func=self.__xml_reader,
            inp=[xml_path],
            Tout=tf.float32,
            name="XML_reader",
        )
        print("COMPLETE")
        
        augmentation_list = []
        
        augment = YOLOv3Augmentation(
            image_size = self.__image_size, 
            translate_range = self.__translate_range, 
            rotation_range = self.__rotation_range,
        )

        if self.__translate_range is not None:
            augmentation_list.append(augment.translation)

        if self.__rotation_range is not None:
            augmentation_list.append(augment.rotation_complex)

        if self.__horizontal_flip:
            augmentation_list.append(augment.horizontal_flip)

        if self.__vertical_flip:
            augmentation_list.append(augment.vertical_flip)

        augmentation_list.append(augment.padding)

        print("AUGMENTATION: ", end="")
        for func in augmentation_list:
            print(f"{func.__name__} -> ", end="")
            image, annotation_list = tf.numpy_function(
                func=func,
                inp=[image, annotation_list],
                Tout=[tf.float32, tf.float32],
                name="Image_" + func.__name__,
            )
        print("COMPLETE")

        print("Generate annotation labels ... ", end="")
        small, medium, large = tf.numpy_function(
            func=self.__generate_annotation_label,
            inp=[annotation_list],
            Tout=[tf.float32, tf.float32, tf.float32],
            name="Annotation_labeling",
        )
        print("COMPLETE")

        image = tf.ensure_shape(image, (self.__image_size[1], self.__image_size[0], 3))
        small = tf.ensure_shape(small, (self.__image_size[1] // 32, self.__image_size[0] // 32, self.__num_anchor, 5 + len(self.__label_list)))
        medium = tf.ensure_shape(medium, (self.__image_size[1] // 16, self.__image_size[0] // 16, self.__num_anchor, 5 + len(self.__label_list)))
        large = tf.ensure_shape(large, (self.__image_size[1] // 8, self.__image_size[0] // 8, self.__num_anchor, 5 + len(self.__label_list)))

        return image, (small, medium, large)

    # Public methods
    def generate_dataset(self, batch_size, drop_reminder=False):
        
        # Check batch_size
        if not isinstance(batch_size, int):
            raise ValueError("Invalid batch_size. It should be an integer.")

        # file path
        image_path = [image for image, _ in self.__input]
        xml_path = [xml for _, xml in self.__input]

        # calculate anchor boxes
        print("Calculating anchor size ... ", end="")

        annotation_list = [self.__xml_reader(path) for path in xml_path]
        self.__anchor_boxes = anchor_module.calculate_anchor(self.__num_anchor, annotation_list)
        print(f"{self.__num_anchor} anchor box{'es' if self.__num_anchor > 1 else ''}: {self.__anchor_boxes}")

        #print("Prepare raw dataset ...", end="")
        dataset = tf.data.Dataset.from_tensor_slices((image_path, xml_path))       
        dataset = dataset.map(self.__preprocessing)
        dataset = dataset.shuffle(buffer_size=len(dataset))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_reminder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset