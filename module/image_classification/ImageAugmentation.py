import sys
sys.dont_write_bytecode = True

import cv2
import random
import numpy as np
import tensorflow as tf

class ImageAugmentation:
    """
    A class for performing image augmentation operations such as horizontal and vertical flips,
    translation, rotation, and padding.

    Attributes:
        image_size (tuple): The target size for the augmented image (width, height).
        translate_range (float, tuple): The translation range for both horizontal and vertical directions.
            If a float, it represents a normalized range. If a tuple, it specifies the range for x and y separately.
        rotation_range (int): The maximum rotation angle for image rotation.
        border_method (str): The method used for image border handling. Should be 'constant' or 'replicate'.

    Methods:
        horizontal_flip(image): Perform horizontal flip on the image with a given probability.
        vertical_flip(image): Perform vertical flip on the image with a given probability.
        translation(image): Apply random translation to the image within the specified range.
        rotation(image): Rotate the image by a random angle within the specified range.
        padding(image): Pad the image to match the target size with a specified border handling method.
    """
    def __init__(self, image_size, translate_range, rotation_range, border_method):
        """
        Initializes the ImageAugmentation instance with specified parameters.

        Parameters:
            image_size (tuple): The target size for the augmented image (width, height).
            translate_range (float, tuple): The translation range for both horizontal and vertical directions.
                If a float, it represents a normalized range. If a tuple, it specifies the range for x and y separately.
            rotation_range (int): The maximum rotation angle for image rotation.
            border_method (str): The method used for image border handling. Should be 'constant' or 'replicate'.
        """
        self.__image_size = image_size
        self.__translate_range = translate_range
        self.__rotation_range = rotation_range
        self.__border_method = border_method

    def __flip(self, image, flip_code):
        image = cv2.flip(image, flip_code)
        image = tf.convert_to_tensor(image) 
        return image
    
    @tf.autograph.experimental.do_not_convert
    def __should_apply(self, probability):
        return random.random() > probability
    
    @tf.autograph.experimental.do_not_convert
    def horizontal_flip(self, image):
        if self.__should_apply(0.5):
            return self.__flip(image, 1)
        return image
    
    @tf.autograph.experimental.do_not_convert
    def vertical_flip(self, image):
        if self.__should_apply(0.5):
            return self.__flip(image, 0)
        return image
    
    @tf.autograph.experimental.do_not_convert
    def translation(self, image):
        if isinstance(self.__translate_range, tuple) and len(self.__translate_range) == 2:
            max_translation_x, max_translation_y = self.__translate_range
        elif isinstance(self.__translate_range, float):
            max_translation_x = max_translation_y = self.__translate_range
        else:
            raise ValueError("Invalid translation format")
        
        image_height, image_width = image.shape[:2]
        
        # Calculate pixel translations based on normalized translations
        translation_x_pixels = int(max_translation_x * image_width)
        translation_y_pixels = int(max_translation_y * image_height)

        # Generate random translations within the pixel limits
        translation_x = np.random.randint(-translation_x_pixels, translation_x_pixels)
        translation_y = np.random.randint(-translation_y_pixels, translation_y_pixels)

        # Translate the image
        M = np.array([
            [1, 0, translation_x], # Horizontal direction
            [0, 1, translation_y], # Vertical direction
        ], dtype=np.float32)

        if self.__border_method == "constant":
            image = cv2.warpAffine(image, M, (image_width, image_height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif self.__border_method == "replicate":
            image = cv2.warpAffine(image, M, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE)
        else:
            raise ValueError("Invalid border method. It should be 'constant' or 'replicate'")

        image = tf.convert_to_tensor(image) 
        return image

    @tf.autograph.experimental.do_not_convert
    def rotation(self, image):
        max_rotation_angle = self.__rotation_range  
        image_height, image_width = image.shape[:2]
        rotation_angle = np.random.randint(-max_rotation_angle, max_rotation_angle)

        # Rotate the image
        M = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), rotation_angle, 1)
        
        if self.__border_method == "constant":
            image = cv2.warpAffine(image, M, (image_width, image_height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif self.__border_method == "replicate":
            image = cv2.warpAffine(image, M, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE)
        else:
            raise ValueError("Invalid border method. It should be 'constant' or 'replicate'")
        
        image = tf.convert_to_tensor(image) 
        
        return image

    @tf.autograph.experimental.do_not_convert
    def padding(self, image):
        
        image_height, image_width = image.shape[:2] # OpenCV format
        target_width, target_height = self.__image_size

        # Calculate the padding sizes for both width and height
        pad_width = max(0, target_width - image_width)
        pad_height = max(0, target_height - image_height)

        # Calculate the padding amounts for top, bottom, left, and right
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Pad the image with zeros (black)
        if self.__border_method == "constant":
            image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        elif self.__border_method == "replicate":
            image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REPLICATE)
        else:
            raise ValueError("Invalid border method. It should be 'constant' or 'replicate'")
        
        image = tf.convert_to_tensor(image) 
        
        return image