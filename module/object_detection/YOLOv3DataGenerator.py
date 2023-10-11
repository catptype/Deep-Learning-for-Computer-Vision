import sys
sys.dont_write_bytecode = True

import cv2
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from .YOLOv3Augmentation import YOLOv3Augmentation
from .YOLOv3Anchor import YOLOv3Anchor as anchor_module

class YOLOv3DataGenerator:
    def __init__(self, 
                 input,
                 annotation_dict,
                 image_size, 
                 num_anchor, 
                 horizontal_flip=False, 
                 vertical_flip=False, 
                 translate_range=None, 
                 rotation_range=None,
                 ):
        
        # Error handler
        self.__validate_input(input)
        self.__validate_image_size(image_size)
        self.__validate_translation(translate_range)

        # Private artibute    
        self.__input = input
        self.__annotation_dict = annotation_dict
        self.__image_size = image_size
        self.__num_anchor = num_anchor
        self.__horizontal_flip = horizontal_flip
        self.__vertical_flip = vertical_flip
        self.__translate_range = translate_range
        self.__rotation_range = rotation_range
        
    # Private methods
    def __validate_input(self, input):
        if not isinstance(input, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in input):
            raise ValueError("Invalid input format")

    def __validate_image_size(self, image_size):
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("Invalid image_size. It should be a tuple (width, height).")

    def __validate_translation(self, translate_range):
        is_valid_tuple = isinstance(translate_range, tuple) and len(translate_range) == 2
        is_valid_float = isinstance(translate_range, float)
        if not (is_valid_tuple or is_valid_float or translate_range is None):
            raise ValueError("Invalid translation format. It should be a tuple (x, y) or a float.")
    
    def __image_reader_cv(self, image_path):
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width, _ = image.shape
        aspect_ratio = original_width / original_height

        # Calculate new image size depend on aspect ratio
        target_width, target_height = self.__image_size
        
        if aspect_ratio == 1: # Perfect square
            new_width = new_height = min(self.__image_size)
        
        elif aspect_ratio > 1: # Landscape orientation (wider than tall)
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        
        else: # Portrait orientation (taller than wide)
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image using interpolation
        image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
        
        # Convert pixel values from int to float by dividing by 255.0
        image = image / 255.0
        image = image.astype("float32")
        
        image = tf.convert_to_tensor(image)
        image_width, image_height = self.__image_size
        image = tf.ensure_shape(image, (image_height, image_width, 3))

        return image
    
    def __image_reader_tf(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(images=image, 
                                size=self.__image_size,
                                preserve_aspect_ratio=True,
                                method="bicubic",
                                antialias=True)
        image = image / 255.0
        return image
    
    def __xml_reader(self, xml_path):
        annotation_list = []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            # Class name or label
            class_name = obj.find('classname').text
            class_idx = self.__annotation_dict[class_name]
            
            # Boundary box positions
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            annotation = (class_idx, xmin, ymin, xmax, ymax)
            annotation_list.append(annotation)
        
        annotation_list = tf.convert_to_tensor(annotation_list)
        annotation_list = tf.ensure_shape(annotation_list, (None, annotation_list.shape[-1]))
        return annotation_list
    
    def __generate_annotation_label(self, annotation_list):

        label_data = []

        for downsampling_scale in [32, 16, 8]: # YOLOv3 always produce 3 scales prediction
            # Calculate grid size for the current scale
            
            col_grid_size = self.__image_size[0] // downsampling_scale #32 width
            row_grid_size = self.__image_size[1] // downsampling_scale #64 height

            # Initialize label data for the current scale
            label_data_shape = (row_grid_size, col_grid_size, self.__num_anchor, 5 + len(self.__annotation_dict))
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
        small = tf.ensure_shape(small, (self.__image_size[1] // 32, self.__image_size[0] // 32, self.__num_anchor, 5 + len(self.__annotation_dict)))
        medium = tf.ensure_shape(medium, (self.__image_size[1] // 16, self.__image_size[0] // 16, self.__num_anchor, 5 + len(self.__annotation_dict)))
        large = tf.ensure_shape(large, (self.__image_size[1] // 8, self.__image_size[0] // 8, self.__num_anchor, 5 + len(self.__annotation_dict)))

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