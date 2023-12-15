import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.metrics import Metric
from sklearn.metrics import precision_recall_curve as pr_curve, auc

class MeanAveragePrecision(Metric):
    """
    Custom metric for computing the Mean Average Precision (mAP) in object detection tasks.

    Inherits from tf.keras.metrics.Metric.

    Parameters:
        num_class (int): Number of classes in the dataset.
        confidence_threshold (float): Confidence threshold for considering detections. Default is 0.2.
        iou_threshold (float): Intersection over Union (IoU) threshold for non-maximum suppression. Default is 0.5.
        max_boxes (int or 'auto'): Maximum number of boxes to consider during non-maximum suppression. Default is 'auto'.
        name (str): Name of the metric. Default is 'mAP'.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        num_class (int): Number of classes in the dataset.
        confidence_threshold (float): Confidence threshold for considering detections.
        iou_threshold (float): Intersection over Union (IoU) threshold for non-maximum suppression.
        max_boxes (int or 'auto'): Maximum number of boxes to consider during non-maximum suppression.

    Private Methods:
        __pre_processing(tensor): Pre-process the input tensor to handle invalid boxes.
        __decoder(tensor): Decode the YOLO tensor into object confidence, box coordinates, and class probabilities.
    
    Public Methods:
        calculate_iou(box1, box2): Calculate the Intersection over Union (IoU) between two bounding boxes.
        update_state(y_true, y_pred, sample_weight=None): Update the state of the metric based on true and predicted values.
        result(): Compute and return the Mean Average Precision (mAP) score.
        get_config(): Get the configuration of the metric.
        from_config(cls, config): Create an instance of the metric from a configuration dictionary.

    Example:
        ```python
        # Example usage
        mAP_metric = MeanAveragePrecision(num_class=20)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mAP_metric])
        ```
    """
    def __init__(self, num_class, confidence_threshold=0.2, iou_threshold=0.5, max_boxes='auto', name='mAP', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.num_class = num_class
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.__total_detect_result = None

    def __pre_processing(self, tensor):
        tensor_np = tensor.numpy()
    
        # Check for invalid boxes in a vectorized manner
        invalid_boxes = np.any((tensor_np[..., 1:5] < 0) | (tensor_np[..., 1:5] > 1), axis=-1)
        
        # Set the first element of each invalid box to -3
        tensor_np[invalid_boxes, 0] = -3

        # Update x and y coordinates for valid boxes
        valid_boxes = ~invalid_boxes
        _, x_indices, y_indices, _ = np.indices(tensor.shape[:-1])
        tensor_np[valid_boxes, 1] += x_indices[valid_boxes]
        tensor_np[valid_boxes, 2] += y_indices[valid_boxes]

        return tf.constant(tensor_np, dtype=tensor.dtype)
    
    def __decoder(self, tensor):
        # Format (Object_confidence, x, y, width, height, class_prob1, class_prob2, ...)
        tensor = self.__pre_processing(tensor)
        batch, x, y, anchor, data = tensor.shape
        tensor = tf.reshape(tensor, [batch, -1, data])
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
        total_detect_result = {}

        # Decode tensor into batch of 1-D shape
        true_obj, true_box, true_class = self.__decoder(y_true)
        pred_obj, pred_box, pred_class = self.__decoder(y_pred)

        # Post-processing to for value range [0,1]
        pred_obj = tf.math.sigmoid(pred_obj)
        pred_class = tf.math.softmax(pred_class)

        # Get all true_box indices
        true_indices = tf.where(true_obj == 1.0)

        # post processing true_indices
        batch_indices, unique_idx = tf.unique(true_indices[:, 0])
        true_indices = tf.dynamic_partition(true_indices[:, 1], unique_idx, len(batch_indices))
        true_indices = [tensor.numpy().tolist() for tensor in true_indices]

        batch_size = y_true.shape[0]
        
        for i in range(batch_size):

            # Apply NMS
            max_boxes = min(len(true_indices[i]), 10) if self.max_boxes == 'auto' else self.max_boxes
            selected_indices = tf.image.non_max_suppression(pred_box[i], pred_obj[i], max_output_size=max_boxes, score_threshold=self.confidence_threshold)
            selected_indices = selected_indices.numpy().tolist()

            # Counting TP
            true_indices_copy = true_indices[i].copy()

            while true_indices_copy:
                true_idx = true_indices_copy.pop(0)
                class_idx = tf.argmax(true_class[i][true_idx])
                pred_idx_list = selected_indices.copy() 

                while pred_idx_list:
                    pred_idx = pred_idx_list.pop(0)
                    if tf.argmax(pred_class[i][pred_idx]) != class_idx:
                        continue

                    if self.calculate_iou(true_box[i][true_idx], pred_box[i][pred_idx]) > self.iou_threshold:
                        class_idx = int(class_idx.numpy())
                        confidence_score = pred_obj[i][pred_idx].numpy()
                        total_detect_result.setdefault(class_idx, []).append((confidence_score, "TP"))
                        selected_indices.remove(pred_idx)

            # Counting FP for remaining boxes
            while selected_indices:
                pred_idx = selected_indices.pop(0)
                class_idx = int(tf.argmax(pred_class[i][pred_idx]).numpy())
                confidence_score = pred_obj[i][pred_idx].numpy()
                total_detect_result.setdefault(class_idx, []).append((confidence_score, "FP"))
        
        self.__total_detect_result = total_detect_result

    @tf.autograph.experimental.do_not_convert
    def result(self):
        ap_per_class = []
        for class_idx in range(self.num_class):
            # dict {class_idx: [(confident, "TP"), (confident, "FP")]}
            detect_result = self.__total_detect_result.get(class_idx, [])
            if detect_result == []:
                ap_per_class.append(0.0)
                continue
            detect_result = sorted(detect_result, key=lambda x: x[0], reverse=True)
        
            confidence_list = []
            tp_fp_list = []
            for item1, item2 in detect_result:
                confidence_list.append(item1)
                tp_fp_list.append(item2 == "TP")
            
            if not any(tp_fp_list):
                ap_per_class.append(0.0)
                continue
            
            # Calculate precision and recall
            precision, recall, _ = pr_curve(tp_fp_list, confidence_list)
            ap_score = auc(recall, precision)
            ap_per_class.append(ap_score)
        
        mAP = sum(ap_per_class) / self.num_class

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