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
    Custom layer implementing a Convolutional Block Attention Module (CBAM).

    Inherits from tf.keras.layers.Layer.

    Parameters:
        ratio (int): Reduction ratio for the channel attention mechanism.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Methods:
        build(input_shape): Build the layer by initializing the necessary components.
        call(input): Apply the layer to the input tensor, performing channel and spatial attention.
        get_config(): Get the configuration of the layer.

    Example:
        ```python
        # Example usage in functional API
        # ... (previous layers)
        x = CBAM_Module(ratio=16)(previous_layer)
        # ... (add other layers)
        ```
    """
    num_instances = 0

    def __init__(self, ratio=16, **kwargs):
        CBAM_Module.num_instances += 1
        layer_name = f"CBAM_Module_{CBAM_Module.num_instances}"
        super(CBAM_Module, self).__init__(name=layer_name)
        self.ratio = ratio

    def build(self, input_shape):
        num_channel = input_shape[-1]
        # Channel_Attention
        ### Shared layers
        self.w0 = Dense(num_channel // self.ratio, activation="relu", kernel_initializer="he_normal")
        self.w1 = Dense(num_channel, kernel_initializer="he_normal")

        # Spatial_Attention
        self.spatial_conv = Conv2D(1, (7, 7), padding="same", activation="sigmoid", kernel_initializer="glorot_normal")
        super().build(input_shape)

    def call(self, input):
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
    
    def get_config(self):
        config = super(CBAM_Module, self).get_config()
        config.update({
            'ratio': self.ratio,
        })
        return config