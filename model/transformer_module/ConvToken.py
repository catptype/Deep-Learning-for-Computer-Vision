import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Layer

class ConvToken(Layer):
    """
    Custom layer to apply a series of convolutional token to the input tensor.

    Inherits from tf.keras.layers.Layer.

    Parameters:
        conv_layer (list): List of integers representing the number of features for each convolutional layer.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Methods:
        build(input_shape): Build the layer by initializing the specified convolutional layers.
        call(input): Apply the layer to the input tensor, performing convolution and max pooling.
        get_config(): Get the configuration of the layer.
        from_config(cls, config): Create an instance of the layer from a configuration dictionary.

    Example:
        ```python
        # Example usage in functional API
        input = Input(shape=(height, width, channel))
        x = ConvToken(conv_layer=[64, 128, 256])(input)
        # ... (add other layers)
        ```
    """
    def __init__(self, conv_layer, **kwargs):
        self.conv_layer = conv_layer
        super(ConvToken, self).__init__(name="ConvToken")
    
    def build(self, input_shape):
        self.conv2d_list = [Conv2D(num_feature, (3,3), padding="same", activation='relu', kernel_initializer="he_normal") for num_feature in self.conv_layer]
        super().build(input_shape)
    
    def call(self, input):
        for idx, _ in enumerate(self.conv_layer):
            if idx == 0:
                x = self.conv2d_list[idx](input)
            else:
                x = self.conv2d_list[idx](x)
            x = MaxPooling2D()(x)
        num_patch = x.shape[1] * x.shape[2]
        output = tf.reshape(x, (-1, num_patch, x.shape[-1]))
        return output
    
    def get_config(self):
        # Return a dictionary with the layer's configuration
        config = super(ConvToken, self).get_config()
        config['conv_layer'] = self.conv_layer
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)