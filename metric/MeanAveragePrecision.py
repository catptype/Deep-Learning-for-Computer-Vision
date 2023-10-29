import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K


class MeanAveragePrecision(Metric):
    def __init__(self, num_class, confidence_threshold=0.2, iou_threshold=0.5, max_boxes=100, name='mAP', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.num_class = num_class
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.true_positives = [self.add_weight(f'true_positives_{i}', initializer='zeros') for i in range(num_class)]
        self.false_positives = [self.add_weight(f'false_positives_{i}', initializer='zeros') for i in range(num_class)]
        self.false_negatives = [self.add_weight(f'false_negatives_{i}', initializer='zeros') for i in range(num_class)]

    def __decoder(self, tensor):
        # Format (Object_confidence, x, y, width, height, class_prob1, class_prob2, ...)
        tensor = tf.reshape(tensor, [-1, tensor.shape[-1]])
        object_score = tensor[..., 0]
        box_coordinate = tensor[..., 1:5]
        class_prob = tensor[..., 5:]
        return object_score, box_coordinate, class_prob

    def calculate_iou(self, box1, box2):
        # Extract coordinates and dimensions from the input tensors
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the coordinates of the intersection rectangle
        x_intersection = tf.maximum(x1, x2)
        y_intersection = tf.maximum(y1, y2)
        w_intersection = tf.maximum(0.0, tf.minimum(x1 + w1, x2 + w2) - x_intersection)
        h_intersection = tf.maximum(0.0, tf.minimum(y1 + h1, y2 + h2) - y_intersection)

        # Calculate the area of intersection and the areas of the two boxes
        area_intersection = w_intersection * h_intersection
        area_box1 = w1 * h1
        area_box2 = w2 * h2

        # Calculate the IoU
        iou = area_intersection / (area_box1 + area_box2 - area_intersection)

        return iou

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):       
        batch_size = y_true.shape[0]
        for i in range(batch_size):
            # Decode tensor into 1-D shape
            true_obj, true_box, true_class = self.__decoder(y_true[i])
            pred_obj, pred_box, pred_class = self.__decoder(y_pred[i])

            # Post-processing to for value range [0,1]
            pred_obj = tf.math.sigmoid(pred_obj)
            pred_class = tf.math.softmax(pred_class)

            true_indices = tf.reshape(tf.where(true_obj == 1.0), [-1])
            true_indices = list(true_indices.numpy())

            # Apply NMS
            selected_indices = tf.image.non_max_suppression(pred_box, pred_obj, max_output_size=self.max_boxes)
            selected_indices = list(selected_indices.numpy())
            
            # Post processing
            selected_indices = [idx for idx in selected_indices if pred_obj[idx] >= self.confidence_threshold]

            for true_idx in true_indices:
                class_idx = tf.argmax(true_class[true_idx])
                for pred_idx in selected_indices:
                    
                    is_same_position = true_idx == pred_idx
                    is_valid_iou = self.calculate_iou(true_box[true_idx], pred_box[pred_idx]) > self.iou_threshold
                    is_correct_class = tf.argmax(pred_class[pred_idx]) == class_idx

                    if is_same_position and is_valid_iou and is_correct_class:
                        self.true_positives[class_idx].assign_add(1)
                    elif is_same_position and is_valid_iou and not is_correct_class:
                        self.false_positives[class_idx].assign_add(1)
                    elif not is_same_position and pred_obj[pred_idx] > true_obj[true_idx]: # true_obj always 0 and pred_obj always > 0
                        self.false_positives[class_idx].assign_add(1)
                    else:
                        self.false_negatives[class_idx].assign_add(1)
            
    def result(self):
        ap_per_class = []
        for class_idx in range(self.num_class):          
            class_tp = self.true_positives[class_idx]
            class_fp = self.false_positives[class_idx]
            class_fn = self.false_negatives[class_idx]

            precision = class_tp / (class_tp + class_fp + K.epsilon())
            recall = class_tp / (class_tp + class_fn + K.epsilon())

            # Calculate AP for the current class
            ap = precision * recall / (precision + recall + K.epsilon())
            ap_per_class.append(ap)

        # Calculate mAP as the mean of APs for all classes
        mAP = tf.reduce_mean(ap_per_class)
        return mAP
    
    def get_config(self):
        config = super(MeanAveragePrecision, self).get_config()
        config.update({
            'num_class': self.num_class, 
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_boxes': self.max_boxes,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)