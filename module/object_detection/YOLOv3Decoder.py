import cv2
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

class Drawing:

    @staticmethod
    def grid(image, row_grid_size, col_grid_size):
        # Extract variables
        image_height, image_width = image.shape[:2]

        # Boundary box setup
        grid_color = (0, 0, 255)  # Blue for grid lines
        grid_thickness = 1

        cell_width = image_width // col_grid_size
        cell_height = image_height // row_grid_size

        # Draw vertical lines
        for i in range(col_grid_size):
            x = i * cell_width
            cv2.line(image, (x, 0), (x, image.shape[0]), grid_color, grid_thickness)

        # Draw horizontal lines
        for i in range(row_grid_size):
            y = i * cell_height
            cv2.line(image, (0, y), (image.shape[1], y), grid_color, grid_thickness)

    @staticmethod
    def bounding_box(image, confidence_score, yxyx_box, class_name):
        # Extract box information
        ymin, xmin, ymax, xmax = yxyx_box.numpy()

        # Boundary box setup
        color = (0, 255, 0)  # Green for the bounding box color
        thickness = 2

        # Draw boundary box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        # Draw text with class name and confidence score
        text = f"{class_name} ({confidence_score:.2f})"
        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    @staticmethod
    def highlight(image, xywh_box, row_size, col_size):
        # Extract variables
        image_height, image_width = image.shape[:2]

        x, y, _, _ = xywh_box.numpy()
        row, col = math.floor(y), math.floor(x)

        # Calculate grid cell coordinates in image
        x_center_image = (col + x - col) * image_width / col_size
        y_center_image = (row + y - row) * image_height / row_size

        # Calculate grid coordinates
        cell_width = image_width // col_size
        cell_height = image_height // row_size
        grid_xmin, grid_ymin = col * cell_width, row * cell_height
        grid_xmax, grid_ymax = (col + 1) * cell_width, (row + 1) * cell_height

        # Draw grid bounding box
        color = (255, 0, 0)  # Red grid lines
        thickness = 1
        cv2.rectangle(image, (grid_xmin, grid_ymin), (grid_xmax, grid_ymax), color, thickness)

        # Draw dots at x_center and y_center
        dot_radius = 2
        dot_color = (255, 0, 0)  # Red dots
        x_center_pixel, y_center_pixel = int(x_center_image), int(y_center_image)
        cv2.circle(image, (x_center_pixel, y_center_pixel), dot_radius, dot_color, -1)

class Calculator:
    @staticmethod
    def padding(image, target_resolution):
        # Calculate the aspect ratio of the original image
        target_height, target_width = target_resolution
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_width / original_height

        # Determine whether to resize based on width or height
        if target_width / aspect_ratio <= target_width: # Resize based on width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else: # Resize based on height
            new_width = int(target_height * aspect_ratio)
            new_height = target_height

        # Calculate padding for both width and height
        pad_width = max(0, target_width - new_width)
        pad_height = max(0, target_height - new_height)

        return (pad_height, pad_width)
    
class FileIO:
    @staticmethod
    def image_preprocessing(image_path, height, width):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(
            image, 
            (height, width),
            preserve_aspect_ratio=True,
            antialias=True,
        )

        # Calculate padding size (top, bottom, left, right)
        padding = Calculator.padding(image, (height, width))

        image = tf.image.resize_with_pad(image, height, width)
        image = image / 255.0
        
        return image, padding
    
    @staticmethod
    def image_reader(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = image.numpy().astype("uint8")
        return image

class YOLOv3Decoder:

    def __init__(self, h5file, confidence_threshold, max_output=100, nms_iou=0.5, nms_confidence=0.2):
        print("Loading model ... ", end="")
        self.model = tf.keras.models.load_model(h5file, compile=False)
        self.confidence_threshold = confidence_threshold
        self.max_output = max_output
        self.nms_iou = nms_iou
        self.nms_confidence = nms_confidence
        print("COMPLETE")
    
    # Private method
    def __decoder(self, tensor, batch=True):
        # Pre-processing
        # Check for invalid boxes and update raw confidence score = -3
        tensor_np = tensor.numpy()
        invalid_boxes = np.any((tensor_np[..., 1:5] < 0) | (tensor_np[..., 1:5] > 1), axis=-1)
        small_boxes = np.any((tensor_np[..., 3:5] < 0.05), axis=-1)
        tensor_np[invalid_boxes | small_boxes, 0] = -3 # Sigmoid(-3) = 0.04

        # Update x_center and y_center to determine the grid coordinate containing the object
        if batch:
            _, row_indices, col_indices, _ = np.indices(tensor.shape[:-1])
        else:
            row_indices, col_indices, _ = np.indices(tensor.shape[:-1])
        tensor_np[..., 1] += col_indices
        tensor_np[..., 2] += row_indices
        tensor = tf.constant(tensor_np, dtype=tensor.dtype)

        # Start decoding
        if batch:
            batch, _, _, _, data = tensor.shape # grid_x, grid_y, anchor
            tensor = tf.reshape(tensor, [batch, -1, data])
        else:
            _, _, _, data = tensor.shape # grid_x, grid_y, anchor
            tensor = tf.reshape(tensor, [-1, data])
        confidence_score = tensor[..., 0]
        box_coordinate = tensor[..., 1:5]
        class_prob = tensor[..., 5:]
        return confidence_score, box_coordinate, class_prob
    
    def __get_detection(self, image, label_list):
        detection_result = []
        # Predic result
        small, medium, large = self.model.predict(tf.expand_dims(image, axis=0))
        scale_list = [tf.constant(scale) for scale in [small, medium, large]]

        batch_size = small.shape[0]

        image = np.clip(image, 0, 1)
        image = (image * 255).astype("uint8")

        for batch_idx in range(batch_size):

            image_height, image_width = image.shape[:2]

            all_obj, all_box, all_class = [], [], []

            # Merging all detection result
            for scale_output in scale_list:
                _, row_size, col_size, _, _ = scale_output.shape
                
                pred_obj, pred_box, pred_class = self.__decoder(scale_output[batch_idx], batch=False)
                pred_obj = tf.math.sigmoid(pred_obj)
                pred_class = tf.math.softmax(pred_class)

                yxyx_box = self.__xywh2yxyx(pred_box, row_size, col_size)

                all_obj.append(pred_obj)
                all_box.append(yxyx_box)
                all_class.append(pred_class)

            all_obj = tf.concat(all_obj, axis=0)
            all_box = tf.concat(all_box, axis=0)
            all_class = tf.concat(all_class, axis=0)

            # Applying non-max suppression
            selected_indices = tf.image.non_max_suppression(
                tf.cast(all_box, dtype=tf.float32), 
                all_obj, 
                max_output_size=self.max_output, 
                iou_threshold=self.nms_iou,
                score_threshold=self.nms_confidence,
            )
            selected_indices = selected_indices.numpy().tolist()

            for idx in selected_indices:
                confidence_score = all_obj[idx].numpy()
                if confidence_score > self.confidence_threshold:
                    class_idx = tf.argmax(all_class[idx])
                    class_name = label_list[class_idx]

                    ymin, xmin, ymax, xmax = all_box[idx].numpy()
                    ymin /= image_height
                    xmin /= image_width
                    ymax /= image_height
                    xmax /= image_width
                    
                    norm_box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
                    norm_box = tf.cast(norm_box, dtype=tf.float32)
                    
                    detection_result.append((confidence_score, norm_box, class_name))

        return detection_result

    def __xywh2yxyx(self, xywh_box, row_size, col_size):
        _, row_image, col_image, _ = self.model.input.shape
        
        # Convert box information from grid coordinates to image coordinates
        x_center = xywh_box[:, 0] * col_image / col_size
        y_center = xywh_box[:, 1] * row_image / row_size
        width  = xywh_box[:, 2] * col_image
        height = xywh_box[:, 3] * row_image

        # Convert from (x,y,w,h) to (y,x,y,x) format
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        yxyx_box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        yxyx_box = tf.cast(yxyx_box, dtype=tf.int32)

        return yxyx_box
        
    # Public method
    def show_detection(self, image_path, label_list, figsize=(5,5)):
        _, model_height, model_width, _ = self.model.input.shape

        image_original = FileIO.image_reader(image_path)
        image_resize, padding = FileIO.image_preprocessing(image_path, model_height, model_width)

        detection_list = self.__get_detection(image_resize, label_list)

        original_height, original_width = image_original.shape[:2]
        pad_height, pad_width = padding
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        plt.figure(figsize=figsize)
        for detection in detection_list:
            confidence_score, box, class_name = detection
            ymin, xmin, ymax, xmax = box.numpy()

            # Convert coordinate after remove padding
            ymin = (ymin * model_height - pad_top) / (model_height - pad_height)
            xmin = (xmin * model_width - pad_left) / (model_width - pad_width)
            ymax = (ymax * model_height - pad_top) / (model_height - pad_height)
            xmax = (xmax * model_width - pad_left) / (model_width - pad_width)

            ymin *= original_height
            xmin *= original_width
            ymax *= original_height
            xmax *= original_width
            
            box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            box = tf.cast(box, tf.int32)
            Drawing.bounding_box(image_original, confidence_score, box, class_name)
        
        plt.title("Detection result", fontsize=20)
        plt.imshow(image_original)                 
        plt.axis("off")
    
    def extract_detection(self, image_path, label_list):
        _, model_height, model_width, _ = self.model.input.shape

        image_original = FileIO.image_reader(image_path)
        image_resize, padding = FileIO.image_preprocessing(image_path, model_height, model_width)

        detection_list = self.__get_detection(image_resize, label_list)

        original_height, original_width = image_original.shape[:2]
        pad_height, pad_width = padding
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        extract_result = []
        for detection in detection_list:
            _, box, _ = detection
            ymin, xmin, ymax, xmax = box.numpy()

            # Convert coordinate after remove padding
            ymin = (ymin * model_height - pad_top) / (model_height - pad_height)
            xmin = (xmin * model_width - pad_left) / (model_width - pad_width)
            ymax = (ymax * model_height - pad_top) / (model_height - pad_height)
            xmax = (xmax * model_width - pad_left) / (model_width - pad_width)

            ymin *= original_height
            xmin *= original_width
            ymax *= original_height
            xmax *= original_width

            box = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            box = tf.cast(box, tf.int32)

            cropped_img = Image.fromarray(image_original).crop(box.numpy())
            extract_result.append(cropped_img)

        return extract_result
            