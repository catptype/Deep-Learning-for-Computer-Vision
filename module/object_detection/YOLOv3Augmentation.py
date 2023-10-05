import sys
sys.dont_write_bytecode = True

import cv2
import random
import numpy as np
import tensorflow as tf

class YOLOv3Augmentation:
    def __init__(self, image_size, translate_range, rotation_range):
        self.__image_size = image_size
        self.__translate_range = translate_range
        self.__rotation_range = rotation_range

    def __flip(self, image, annotation_list, flip_code):
        image = cv2.flip(image, flip_code)
        updated_annotation_list = []

        for annotation in annotation_list:
            for class_idx, xmin, ymin, xmax, ymax in [annotation]:
                if flip_code == 1:  # Horizontal flip
                    xmin, xmax = 1 - xmax, 1 - xmin
                else:  # Vertical flip
                    ymin, ymax = 1 - ymax, 1 - ymin

                new_annotation = (class_idx, xmin, ymin, xmax, ymax)
                updated_annotation_list.append(new_annotation)

        image = tf.convert_to_tensor(image)
        updated_annotation_list = tf.convert_to_tensor(updated_annotation_list)
        
        return image, updated_annotation_list

    @tf.autograph.experimental.do_not_convert
    def horizontal_flip(self, image, annotation_list):
        if random.random() < 0.5:
            return image, annotation_list
        else:
            return self.__flip(image, annotation_list, 1)
    
    @tf.autograph.experimental.do_not_convert
    def vertical_flip(self, image, annotation_list):
        if random.random() < 0.5:
            return image, annotation_list
        else:
            return self.__flip(image, annotation_list, 0)
    
    @tf.autograph.experimental.do_not_convert
    def translation(self, image, annotation_list):

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
            [1, 0, translation_x], # Translate image in horizontal direction
            [0, 1, translation_y], # Translate image in vertical direction
        ], dtype=np.float32)
        image = cv2.warpAffine(image, M, (image_width, image_height))

        # Update annotation list
        updated_annotation_list = []

        for annotation in annotation_list:
            for class_idx, xmin, ymin, xmax, ymax in [annotation]:

                # Translate the bounding box coordinates
                xmin += translation_x / image_width
                xmax += translation_x / image_width
                ymin += translation_y / image_height
                ymax += translation_y / image_height

                # Checking position
                xmin = max(0, xmin)
                xmax = min(1, xmax)
                ymin = max(0, ymin)
                ymax = min(1, ymax)
                
                # Create a new annotation tuple
                new_annotation = (class_idx, xmin, ymin, xmax, ymax)
                updated_annotation_list.append(new_annotation)

        image = tf.convert_to_tensor(image)
        updated_annotation_list = tf.convert_to_tensor(updated_annotation_list)

        return image, updated_annotation_list

    @tf.autograph.experimental.do_not_convert
    def rotation_complex(self, image, annotation_list): # more computation and more accurate when high rotation
        max_rotation_angle = self.__rotation_range  

        image_height, image_width = image.shape[:2]

        rotation_angle = np.random.randint(-max_rotation_angle, max_rotation_angle)

        # Rotate the image
        M = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), rotation_angle, 1)
        image = cv2.warpAffine(image, M, (image_width, image_height))

        # Update annotation list
        updated_annotation_list = []

        for annotation in annotation_list:
            for class_idx, xmin, ymin, xmax, ymax in [annotation]:

                # Convert normalized bounding box coordinates to pixel coordinates
                xmin *= image_width
                ymin *= image_height
                xmax *= image_width
                ymax *= image_height

                # Create an array of points representing the corners of the bounding box
                points = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], dtype=np.float32)
                points = np.column_stack((points, np.ones(4, dtype=np.float32)))

                # Rotate the points using matrix multiplication with M
                rotated_points = (np.dot(M, points.T).T).astype(np.float32)

                # Find the rotated bounding rectangle
                rotated_rect = cv2.minAreaRect(rotated_points[:, :2])

                # Get the rotated bounding box vertices
                rotated_box = cv2.boxPoints(rotated_rect).astype(int)

                # Find the new coordinates of the rotated bounding box
                rotated_xmin = min(rotated_box[:, 0]) / image_width
                rotated_ymin = min(rotated_box[:, 1]) / image_height
                rotated_xmax = max(rotated_box[:, 0]) / image_width
                rotated_ymax = max(rotated_box[:, 1]) / image_height

                # Create a new annotation tuple
                new_annotation = (class_idx, rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax)
                updated_annotation_list.append(new_annotation)

        image = tf.convert_to_tensor(image)
        updated_annotation_list = tf.convert_to_tensor(updated_annotation_list)
        
        return image, updated_annotation_list

    @tf.autograph.experimental.do_not_convert
    def rotation(self, image, annotation_list): # not accurate when high rotation
        # Define maximum allowed rotation angle in degrees
        max_rotation_angle = self.__rotation_range  

        image_height, image_width = image.shape[:2]
        rotation_angle = np.random.randint(-max_rotation_angle, max_rotation_angle)

        # Rotate the image
        M = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), rotation_angle, 1)
        image = cv2.warpAffine(image, M, (image_width, image_height))

        # Update annotation list
        updated_annotation_list = []

        for annotation in annotation_list:
            for class_idx, xmin, ymin, xmax, ymax in [annotation]:

                # Convert normalized bounding box coordinates to pixel coordinates
                xmin *= image_width
                ymin *= image_height
                xmax *= image_width
                ymax *= image_height

                # Create an array of points representing the corners of the bounding box
                points = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

                # Add a column of ones to the points array
                points_homogeneous = np.column_stack((points, np.ones(4)))

                # Rotate the points using matrix multiplication with M
                rotated_points = np.dot(M, points_homogeneous.T).T

                # Find the new coordinates of the rotated bounding box
                rotated_xmin = min(rotated_points[:, 0])
                rotated_ymin = min(rotated_points[:, 1])
                rotated_xmax = max(rotated_points[:, 0])
                rotated_ymax = max(rotated_points[:, 1])

                # Convert rotated coordinates back to normalized coordinates
                rotated_xmin /= image_width
                rotated_ymin /= image_height
                rotated_xmax /= image_width
                rotated_ymax /= image_height

                # Create a new annotation tuple
                new_annotation = (class_idx, rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax)
                updated_annotation_list.append(new_annotation)

        image = tf.convert_to_tensor(image)
        updated_annotation_list = tf.convert_to_tensor(updated_annotation_list)
        
        return image, updated_annotation_list

    @tf.autograph.experimental.do_not_convert
    def padding(self, image, annotation_list):
        
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
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

        # Update annotation list
        updated_annotation_list = []

        for annotation in annotation_list:
            for class_idx, xmin, ymin, xmax, ymax in [annotation]:
                # Adjust the bounding box coordinates for padding and normalize them
                image_height_padded, image_width_padded = image.shape[:2]
                xmin = (xmin * image_width + left_pad) / image_width_padded
                ymin = (ymin * image_height + top_pad) / image_height_padded
                xmax = (xmax * image_width + left_pad) / image_width_padded
                ymax = (ymax * image_height + top_pad) / image_height_padded

                # Create a new annotation tuple
                new_annotation = (class_idx, xmin, ymin, xmax, ymax)
                updated_annotation_list.append(new_annotation)
        
        image = tf.convert_to_tensor(image)
        updated_annotation_list = tf.convert_to_tensor(updated_annotation_list)
        
        return image, updated_annotation_list