import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Layer

class ClassToken(Layer):
    """
    Custom layer to add a learnable class token to the input tensor.

    Inherits from tf.keras.layers.Layer.

    Methods:
        build(input_shape): Build the layer by initializing the learnable class token.
        call(input): Apply the layer to the input tensor, adding the class token.
        get_config(): Get the configuration of the layer.
        from_config(cls, config): Create an instance of the layer from a configuration dictionary.

    Example:
        ```python
        # Example usage in functional API
        input = Input(shape=(height, width, channels))
        x = ClassToken()(input)
        # ... (add other layers)
        ```
    """
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(name="ClassToken")
    
    def build(self, input_shape):
        init_w = tf.random_normal_initializer()
        self.init_weight = tf.Variable(
            initial_value = init_w(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True,
        )
        super().build(input_shape)
    
    def call(self, input):
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