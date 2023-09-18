import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K


class F1ScoreMetric(Metric):
    """
    Custom metric for computing the F1 score during model training and evaluation.

    This metric calculates the F1 score, which is the harmonic mean of precision and recall.
    It is suitable for binary classification tasks.

    Args:
        name (str): Name of the metric (default is 'f1_score').
        **kwargs: Additional keyword arguments to be passed to the Metric class.

    Attributes:
        TP (tf.Variable): True positives count.
        FP (tf.Variable): False positives count.
        FN (tf.Variable): False negatives count.

    Methods:
        update_state(y_true, y_pred, sample_weight=None):
            Update the metric's internal counts based on true labels and predicted probabilities.

        result():
            Calculate and return the F1 score based on the internal counts.

    Example:
        To use this metric during model compilation, you can include it in the metrics list:

        ```python
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1ScoreMetric()])
        ```

    Note:
        This metric assumes binary classification and considers a threshold of 0.5 for converting
        predicted probabilities to binary predictions.
    """
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.TP = self.add_weight(name='true_positives', initializer='zeros')
        self.FP = self.add_weight(name='false_positives', initializer='zeros')
        self.FN = self.add_weight(name='false_negatives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Assuming binary classification
        
        true_pos  = tf.reduce_sum(y_true * y_pred)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.TP.assign_add(true_pos)
        self.FP.assign_add(false_pos)
        self.FN.assign_add(false_neg)

    def result(self):
        precision = self.TP / (self.TP + self.FP + K.epsilon())
        recall    = self.TP / (self.TP + self.FN + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1