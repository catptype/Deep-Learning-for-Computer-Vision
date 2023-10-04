import sys
sys.dont_write_bytecode = True

import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET

class YOLOv3FileReader:

    @staticmethod
    def image_cv(image_path, image_size):
        
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the original image dimensions
        original_height, original_width, _ = image.shape

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Calculate the new dimensions while maintaining the aspect ratio
        target_width, target_height = image_size
        
        if aspect_ratio == 1:
            # Perfect square shape
            new_width = min(image_size)
            new_height = min(image_size)   
        elif aspect_ratio > 1:
            # Landscape orientation (wider than tall)
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Portrait or square orientation (taller than wide)
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image using interpolation
        image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
        image = image.astype("float32")
        
        # Convert pixel values from int to float by dividing by 255.0
        image = image / 255.0
        
        return tf.convert_to_tensor(image)
    
    @staticmethod
    def image_tf(image_path, image_size):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(image, image_size,
                                preserve_aspect_ratio=True,
                                method="bicubic",
                                antialias=True)
        image = image / 255.0
        return image
    
    @staticmethod
    def xml(xml_path, annotation_dict):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotation_list = []

        for obj in root.findall('object'):
            # Class name or label
            class_name = obj.find('classname').text
            class_idx = annotation_dict[class_name]
            
            # Boundary box positions
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            annotation = (class_idx, xmin, ymin, xmax, ymax)

            annotation_list.append(annotation)
        
        return tf.convert_to_tensor(annotation_list)