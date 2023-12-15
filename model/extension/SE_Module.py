from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Layer,
    Multiply,
    ReLU,
    Reshape,
)

class SE_Module(Layer):
    """
    Squeeze-and-Excitation (SE) Module implementation as a custom Keras layer.
    This module performs channel-wise feature recalibration to enhance
    important features and suppress less important ones.
    """
    num_instances = 0

    def __init__(self, ratio=16, **kwargs):
        """
        Initialize the SE_Module.

        Parameters:
            ratio (int): Reduction ratio for the dense layers.
        """
        SE_Module.num_instances += 1
        layer_name = f"SE_module_{SE_Module.num_instances}"
        super(SE_Module, self).__init__(name=layer_name)
        self.ratio = ratio
    
    def build(self, input_shape):
        """
        Build the SE_Module by creating necessary layers.

        Parameters:
            input_shape (tuple): Shape of the input tensor.
        """
        self.num_channels = input_shape[-1]

        # Prepare layers which have learnable parameters
        self.Dense1 = Dense(self.num_channels // self.ratio, activation="relu")
        self.Dense2 = Dense(self.num_channels, activation="sigmoid")
        self.Reshape = Reshape((1, 1, self.num_channels))
        self.Conv2D = Conv2D(self.num_channels, (1, 1), padding="same", kernel_initializer="he_normal")
        self.BatchNorm = BatchNormalization()

        super().build(input_shape)

    def call(self, input, identity):
        """
        Perform the forward pass of the SE_Module.

        Parameters:
            input (Tensor): Input tensor to the module.
            identity (Tensor): Residual connection for the module.

        Returns:
            Tensor: Output tensor after applying the SE_Module.
        """
        # Squeeze: Global average pooling across the spatial dimensions
        x = GlobalAveragePooling2D()(input)

        # Excitation: Two dense layers
        x = self.Dense1(x)
        x = self.Dense2(x)

        # Reshape to (batch_size, 1, 1, num_channels)
        x = self.Reshape(x)

        # Scale the input tensor with the computed channel-wise scaling factors
        x = Multiply()([input, x])

        # Adjust the dimensions of the identity tensor if needed
        if x.shape[-1] != identity.shape[-1]:
            identity = self.Conv2D(identity)
            identity = self.BatchNorm(identity)
            identity = ReLU()(identity)

        # Element-wise addition of the scaled input and the identity
        x = Add()([identity, x])

        return x
    
    def get_config(self):
        config = super(SE_Module, self).get_config()
        config.update({'ratio': self.ratio})
        return config
