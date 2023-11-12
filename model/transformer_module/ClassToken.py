import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Layer,
)


class ClassToken(Layer):
    """
    The ClassToken layer.

    This layer adds a class token to the input tensor. The class token is a
    learnable parameter used in Vision Transformers to represent global information.

    """
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(name="ClassToken")
    
    def build(self, input_shape):
        """
        Builds the ClassToken layer by initializing the class token weight.

        Args:
            input_shape (tuple): The shape of the input tensor.

        """
        init_w = tf.random_normal_initializer()
        self.init_weight = tf.Variable(
            initial_value = init_w(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True,
        )
        super().build(input_shape)
    
    def call(self, input):
        """
        Adds the class token to the input tensor.

        Args:
            input (tf.Tensor): The input tensor to which the class token is added.

        Returns:
            tf.Tensor: The input tensor with the class token added as the first element along the axis 1.
        """
        c_token = tf.broadcast_to(self.init_weight, [tf.shape(input)[0], 1, self.init_weight.shape[-1]])
        c_token = tf.cast(c_token, dtype=input.dtype)

        output = Concatenate(axis=1)([c_token, input])
        return output
    
    def get_config(self):
        config = super(ClassToken, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)