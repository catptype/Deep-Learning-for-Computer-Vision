import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Layer,
)

class ConvToken(Layer):
    """
    ConvToken - A custom Keras layer for creating convolutional token embeddings.

    This class defines a custom Keras layer that takes a list of convolutional layer configurations
    and applies them sequentially to the input tensor. Each convolutional layer is followed by
    max-pooling, and the final output is a reshaped tensor representing token embeddings.

    Attributes:
        conv_layer (list): A list of integers representing the number of features for each
                        convolutional layer.

    """
    def __init__(self, conv_layer):
        """
        Constructor method for initializing the ConvToken layer.

        Args:
            conv_layer (list): A list of integers representing the number of features
                               for each convolutional layer.
        """
        self.conv_layer = conv_layer
        super(ConvToken, self).__init__(name="ConvToken")
    
    def build(self, input_shape):
        """
        Method for building the ConvToken layer with the specified input shape.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Note:
            This method creates a list of Conv2D layers based on the configuration provided
            in the `conv_layer` attribute.

        """
        self.conv2d_list = [Conv2D(num_feature, (3,3), padding="same", activation='relu', kernel_initializer="he_normal") for num_feature in self.conv_layer]
        super().build(input_shape)
    
    def call(self, input):
        """
        Method for performing the forward pass of the ConvToken layer.

        Args:
            input (tensor): The input tensor.

        Returns:
            output (tensor): The output tensor after applying convolutional layers and max-pooling.
        """
        for idx, _ in enumerate(self.conv_layer):
            if idx == 0:
                x = self.conv2d_list[idx](input)
            else:
                x = self.conv2d_list[idx](x)
            x = MaxPooling2D()(x)
        num_patch = x.shape[1] * x.shape[2]
        output = tf.reshape(x, (-1, num_patch, x.shape[-1]))
        return output