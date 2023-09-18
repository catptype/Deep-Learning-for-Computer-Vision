"""
DenseNet Models
~~~~~~~~~~~~~~~~

This module defines a set of DenseNet architectures for image classification tasks using TensorFlow and Keras.

Classes:
    - DenseNetModel: Base class for DenseNet architectures.
    - DenseNet121: Implementation of DenseNet-121 architecture.
    - DenseNet169: Implementation of DenseNet-169 architecture.
    - DenseNet201: Implementation of DenseNet-201 architecture.
    - DenseNet264: Implementation of DenseNet-264 architecture.

Note: To use these models, TensorFlow and Keras must be installed.
"""

import sys
sys.dont_write_bytecode = True

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
    
    Attributes:
        - growth_rate: The growth rate for the DenseNet model.
    
    Methods:
        - init_block(input): Initializes the initial block of the DenseNet model.
        - BN_ReLU_Conv(input, num_feature, kernel): Applies BatchNormalization, ReLU activation, and Convolution layers.
        - Dense_block(input, num_feature, num_layer): Creates a dense block with a given number of layers.
        - Transit_block(input): Creates a transition block with AveragePooling.
    """
    def __init__(self, image_size, num_classes, growth_rate):
        """
        Initializes the DenseNet model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - **kwargs: Additional keyword arguments.
        """
        self.growth_rate = growth_rate
        super().__init__(image_size=image_size, num_classes=num_classes)

    def init_block(self, input):
        x = Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        return x

    def BN_ReLU_Conv(self, input, num_feature, kernel=3):
        x = BatchNormalization()(input)
        x = ReLU()(x)
        x = Conv2D(
            num_feature,
            (kernel, kernel),
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        return x

    def Dense_block(self, input, num_feature, num_layer):
        for i in range(num_layer):
            input_tensor = input if i == 0 else x
            x = self.BN_ReLU_Conv(input_tensor, num_feature * 4, kernel=1)
            x = self.BN_ReLU_Conv(x, num_feature)
            x = Concatenate()([x, input_tensor])
        return x

    def Transit_block(self, input):
        x = self.BN_ReLU_Conv(input, input.shape[-1] // 2, kernel=1)
        x = AveragePooling2D(strides=2)(x)
        return x


class DenseNet121(DenseNetModel):
    """
    Implementation of DenseNet-121 architecture.
    """
    def __init__(self, image_size, num_classes, growth_rate=32):
        """
        Initializes the DenseNet-121 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, growth_rate=growth_rate)

    def build_model(self):
        """
        Builds the DenseNet-121 model.
        
        Returns:
            - model: The built Keras model.
        """
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
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet121_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class DenseNet169(DenseNetModel):
    """
    Implementation of DenseNet-169 architecture.
    """
    def __init__(self, image_size, num_classes, growth_rate=32):
        """
        Initializes the DenseNet-169 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, growth_rate=growth_rate)

    def build_model(self):
        """
        Builds the DenseNet-169 model.
        
        Returns:
            - model: The built Keras model.
        """
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
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet169_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class DenseNet201(DenseNetModel):
    """
    Implementation of DenseNet-201 architecture.
    """    
    def __init__(self, image_size, num_classes, growth_rate=32):
        """
        Initializes the DenseNet-169 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, growth_rate=growth_rate)

    def build_model(self):
        """
        Builds the DenseNet-201 model.
        
        Returns:
            - model: The built Keras model.
        """
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
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet201_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class DenseNet264(DenseNetModel):
    """
    Implementation of DenseNet-264 architecture.
    """    
    def __init__(self, image_size, num_classes, growth_rate=32):
        """
        Initializes the DenseNet-169 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - growth_rate (int): The growth rate for the model.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, growth_rate=growth_rate)

    def build_model(self):
        """
        Builds the DenseNet-264 model.
        
        Returns:
            - model: The built Keras model.
        """
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
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"DenseNet264_k{self.growth_rate}_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model
