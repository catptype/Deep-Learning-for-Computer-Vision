import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPool2D,
    Layer,
    Multiply,
)


class CBAM_Module(Layer):
    """
    Implementation of the Convolutional Block Attention Module (CBAM).
    
    This class defines a custom layer CBAM_Module that implements a mechanism for capturing both channel-wise and 
    spatial-wise attention in a convolutional neural network. CBAM is used to enhance the representational power of 
    the network by selectively emphasizing important features.
    
    Attributes:
        num_instances (int): Keeps track of the number of CBAM instances created.
    
    Args:
        ratio (int): The reduction ratio used in the channel attention mechanism.
    """
    num_instances = 0

    def __init__(self, ratio=16):
        """
        Initializes a CBAM_Module instance.

        Args:
            ratio (int): The reduction ratio used in the channel attention mechanism.
        """
        self.ratio = ratio
        CBAM_Module.num_instances += 1
        layer_name = f"CBAM_Module_{CBAM_Module.num_instances}"
        super(CBAM_Module, self).__init__(name=layer_name)

    def build(self, input_shape):
        """
        Builds the CBAM_Module layer.

        This method is responsible for creating the necessary layers and parameters used in the CBAM module.
        
        Args:
            input_shape (tuple): The shape of the input tensor to the CBAM_Module layer.
        """
        num_channel = input_shape[-1]
        # Channel_Attention
        ### Shared layers
        self.w0 = Dense(num_channel // self.ratio, activation="relu", kernel_initializer="he_normal")
        self.w1 = Dense(num_channel, kernel_initializer="he_normal")

        # Spatial_Attention
        self.spatial_conv = Conv2D(1, (7, 7), padding="same", activation="sigmoid", kernel_initializer="glorot_normal")
        super().build(input_shape)

    def call(self, input):
        """
        Defines the forward pass of the CBAM_Module layer.

        Args:
            input (tf.Tensor): The input tensor to the CBAM_Module layer.

        Returns:
            tf.Tensor: The output tensor after applying channel-wise and spatial-wise attention.
        """
        # Channel_Attention
        ### Global Max Pool
        GMP_pool = GlobalMaxPool2D()(input)
        GMP_pool = self.w0(GMP_pool)
        GMP_pool = self.w1(GMP_pool)

        ### Global Average Pool
        GAP_pool = GlobalAveragePooling2D()(input)
        GAP_pool = self.w0(GAP_pool)
        GAP_pool = self.w1(GAP_pool)

        channel_attention = Add()([GMP_pool, GAP_pool])
        channel_attention = Activation("sigmoid")(channel_attention)
        channel_attention = Multiply()([input, channel_attention])

        # Spatial_Attention
        ### Max pool
        max_pool = tf.reduce_max(channel_attention, axis=-1)
        max_pool = tf.expand_dims(max_pool, axis=-1)

        ### Average pool
        avg_pool = tf.reduce_mean(channel_attention, axis=-1)
        avg_pool = tf.expand_dims(avg_pool, axis=-1)

        spatial_attention = Concatenate()([max_pool, avg_pool])
        spatial_attention = self.spatial_conv(spatial_attention)

        output = Multiply()([channel_attention, spatial_attention])
        return output
