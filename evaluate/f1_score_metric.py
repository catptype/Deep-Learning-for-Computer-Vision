import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from sklearn.metrics import f1_score

def f1_score_metric(y_true, y_pred):
    """
    Create a custom F1 score metric for TensorFlow models that can be used during model compilation.

    Returns:
        callable: Custom F1 score metric function.

    Example:
        To use this custom F1 score metric in model compilation, you can do the following:

        ```python
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score_metric])
        ```
    """
    y_pred = tf.round(y_pred)  # Convert predictions to binary values (0 or 1)
    f1 = tf.py_function(f1_score, (y_true, y_pred), tf.float32)
    return f1