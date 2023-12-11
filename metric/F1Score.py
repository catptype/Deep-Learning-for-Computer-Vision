import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K

class F1Score(Metric):
    """
    Custom metric for computing the F1 score in classification tasks.

    Inherits from tf.keras.metrics.Metric.

    Parameters:
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        true_positives (tf.Variable): Running total of true positives.
        false_positives (tf.Variable): Running total of false positives.
        false_negatives (tf.Variable): Running total of false negatives.

    Methods:
        update_state(y_true, y_pred, sample_weight=None): Update the state of the metric based on true and predicted values.
        result(): Compute and return the current F1 score.
        get_config(): Get the configuration of the metric.
        from_config(cls, config): Create an instance of the metric from a configuration dictionary.

    Example:
        ```python
        # Example usage
        f1_metric = F1Score()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_metric])
        ```
    """
    def __init__(self, **kwargs):
        super(F1Score, self).__init__(name='f1', **kwargs)
        self.true_positives = self.add_weight('true_positives', initializer='zeros')
        self.false_positives = self.add_weight('false_positives', initializer='zeros')
        self.false_negatives = self.add_weight('false_negatives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Assuming binary classification
        
        true_pos  = tf.reduce_sum(y_true * y_pred)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)
        self.false_negatives.assign_add(false_neg)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall    = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1
    
    def get_config(self):
        config = super(F1Score, self).get_config()
        config.update({
            'true_positives': self.true_positives.numpy(),
            'false_positives': self.false_positives.numpy(),
            'false_negatives': self.false_negatives.numpy(),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)