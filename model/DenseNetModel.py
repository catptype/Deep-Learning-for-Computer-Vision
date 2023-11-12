import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel


class DenseNetModel(DeepLearningModel):
    """
    Base class for DenseNet architectures.

    This class serves as the base for implementing various DenseNet architectures.

    Parameters:
        image_size (int): The input image size.
        num_class (int): The number of output classes for classification.
        growth_rate (int): The growth rate of the DenseNet model.

    Methods:
        init_block(input): Create an initial block of the DenseNet model.
        BN_ReLU_Conv(input, num_feature, kernel=3): Apply a BN-ReLU-Conv layer with specified parameters.
        Dense_block(input, num_feature, num_layer): Create a dense block with the given number of layers.
        Transit_block(input): Create a transition block to reduce spatial dimensions.

    """
    def __init__(self, image_size, num_class, growth_rate):
        self.image_size = image_size
        self.num_class = num_class
        self.growth_rate = growth_rate
        super().__init__()

    def init_block(self, input):
        """
        Create the initial block of the DenseNet model.

        Parameters:
            input: Input tensor for the initial block.

        Returns:
            TensorFlow tensor representing the output of the initial block.
        """
        x = Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        return x

    def BN_ReLU_Conv(self, input, num_feature, kernel=3):
        """
        Apply a BN-ReLU-Conv layer with specified parameters.

        Parameters:
            input: Input tensor for the layer.
            num_feature (int): The number of output features.
            kernel (int): The kernel size for the convolution.

        Returns:
            TensorFlow tensor representing the output of the BN-ReLU-Conv layer.
        """
        x = BatchNormalization()(input)
        x = ReLU()(x)
        x = Conv2D(num_feature, (kernel, kernel), padding="same", kernel_initializer="he_normal")(x)
        return x

    def Dense_block(self, input, num_feature, num_layer):
        """
        Create a dense block with the given number of layers.

        Parameters:
            input: Input tensor for the dense block.
            num_feature (int): The number of feature maps.
            num_layer (int): The number of layers in the dense block.

        Returns:
            TensorFlow tensor representing the output of the dense block.
        """
        for i in range(num_layer):
            input_tensor = input if i == 0 else x
            x = self.BN_ReLU_Conv(input_tensor, num_feature * 4, kernel=1)
            x = self.BN_ReLU_Conv(x, num_feature)
            x = Concatenate()([x, input_tensor])
        return x

    def Transit_block(self, input):
        """
        Create a transition block to reduce spatial dimensions.

        Parameters:
            input: Input tensor for the transition block.

        Returns:
            TensorFlow tensor representing the output of the transition block.
        """
        x = self.BN_ReLU_Conv(input, input.shape[-1] // 2, kernel=1)
        x = AveragePooling2D(strides=2)(x)
        return x


class DenseNet121(DenseNetModel):
    """
    Implementation of DenseNet-121 architecture.
    """
    def __init__(self, image_size, num_class, growth_rate=32):
        """
        Initializes the DenseNet-121 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
            growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_class=num_class, growth_rate=growth_rate)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
        x = self.init_block(input)

        # Dense block
        for idx, config in enumerate([6, 12, 24, 6]):
            x = self.Dense_block(x, self.growth_rate, config)
            if idx != 3:  # ignore transit block at final
                x = self.Transit_block(x)

        # Output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet121_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )

        return model


class DenseNet169(DenseNetModel):
    """
    Implementation of DenseNet-169 architecture.
    """
    def __init__(self, image_size, num_class, growth_rate=32):
        """
        Initializes the DenseNet-169 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
            growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_class=num_class, growth_rate=growth_rate)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
        x = self.init_block(input)

        # Dense block
        for idx, config in enumerate([6, 12, 32, 32]):
            x = self.Dense_block(x, self.growth_rate, config)
            if idx != 3:  # ignore transit block at final
                x = self.Transit_block(x)

        # Output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet169_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )

        return model


class DenseNet201(DenseNetModel):
    """
    Implementation of DenseNet-201 architecture.
    """    
    def __init__(self, image_size, num_class, growth_rate=32):
        """
        Initializes the DenseNet-201 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
            growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_class=num_class, growth_rate=growth_rate)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
        x = self.init_block(input)

        # Dense block
        for idx, config in enumerate([6, 12, 48, 32]):
            x = self.Dense_block(x, self.growth_rate, config)
            if idx != 3:  # ignore transit block at final
                x = self.Transit_block(x)

        # Output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet201_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )

        return model


class DenseNet264(DenseNetModel):
    """
    Implementation of DenseNet-264 architecture.
    """    
    def __init__(self, image_size, num_class, growth_rate=32):
        """
        Initializes the DenseNet-264 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
            growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_class=num_class, growth_rate=growth_rate)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
        x = self.init_block(input)

        # Dense block
        for idx, config in enumerate([6, 12, 64, 48]):
            x = self.Dense_block(x, self.growth_rate, config)
            if idx != 3:  # ignore transit block at final
                x = self.Transit_block(x)

        # Output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet264_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )

        return model
