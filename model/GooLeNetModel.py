"""
Inception v1 Models
~~~~~~~~~~~~~~~~~~~

This module defines a set of Inception v1 architectures for image classification tasks using TensorFlow and Keras.

Classes:
    - GooLeNetModel: Base class for Inception v1 architectures.
    - Inception_v1: Implementation of the original Inception v1 architecture.
    - Inception_v1_BN: Implementation of Inception v1 architecture with Batch Normalization.

Note: To use these models, TensorFlow and Keras must be installed.

Author: [Your Name]
Date: [Date Created]
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
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel


class GooLeNetModel(DeepLearningModel):
    """
    Base class for Inception v1 architectures.
    
    Methods:
        - Conv2D_block(input, num_feature, kernel, strides, use_bn): Creates a Convolutional Block.
        - init_block(input, use_bn): Creates the initial block of the architecture.
        - inception_module(input, num_feature_list, use_bn, downsampler): Creates an Inception module.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the Inception v1 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def Conv2D_block(self, input, num_feature, kernel=3, strides=1, use_bn=False):
        x = Conv2D(num_feature, (kernel, kernel), strides=strides, padding="same", kernel_initializer="he_normal")(input)
        if use_bn:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def init_block(self, input, use_bn=False):
        x = self.Conv2D_block(input, 64, kernel=7, strides=2, use_bn=use_bn)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        x = self.Conv2D_block(x, 64, kernel=1, use_bn=use_bn)
        x = self.Conv2D_block(x, 192, kernel=3, use_bn=use_bn)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        return x

    def inception_module(self, input, num_feature_list, use_bn=False, downsampler=False):
        assert (len(num_feature_list) == 6), f"num feature must contain 6 numbers: len = {len(num_feature_list)}"

        conv1x1 = self.Conv2D_block(input, num_feature_list[0], kernel=1, use_bn=use_bn)

        conv3x3 = self.Conv2D_block(input, num_feature_list[1], kernel=1, use_bn=use_bn)
        conv3x3 = self.Conv2D_block(conv3x3, num_feature_list[2], kernel=3, use_bn=use_bn)

        conv5x5 = self.Conv2D_block(input, num_feature_list[3], kernel=1, use_bn=use_bn)
        conv5x5 = self.Conv2D_block(conv5x5, num_feature_list[4], kernel=5, use_bn=use_bn)

        pool_proj = MaxPooling2D((3, 3), strides=1, padding="same")(input)
        pool_proj = self.Conv2D_block(pool_proj, num_feature_list[5], kernel=1, use_bn=use_bn)

        output = Concatenate()([conv1x1, conv3x3, conv5x5, pool_proj])
        if downsampler:
            output = MaxPooling2D((3, 3), strides=2, padding="same")(output)
        return output


class Inception_v1(GooLeNetModel):
    """
    Implementation of the original Inception v1 architecture.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the Inception v1 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the Inception v1 model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input)

        # Stage 1
        x = self.inception_module(x, [64, 96, 128, 16, 32, 32])
        x = self.inception_module(x, [128, 128, 192, 32, 96, 64], downsampler=True)

        # Stage 2
        x = self.inception_module(x, [192, 96, 208, 16, 48, 64])
        x = self.inception_module(x, [160, 112, 224, 24, 64, 64])
        x = self.inception_module(x, [128, 128, 256, 24, 64, 64])
        x = self.inception_module(x, [112, 144, 288, 32, 64, 64])
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128], downsampler=True)

        # Stage 3
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128])
        x = self.inception_module(x, [384, 192, 384, 48, 128, 128])

        # Output
        output = AveragePooling2D((7, 7), strides=1)(x)
        output = Flatten()(output)
        output = Dropout(0.4)(output)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(output)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Inception_v1_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class Inception_v1_BN(GooLeNetModel):
    """
    Implementation of Inception v1 architecture with Batch Normalization.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the Inception v1 model with Batch Normalization.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the Inception v1 model with Batch Normalization.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input, use_bn=True)

        # Stage 1
        x = self.inception_module(x, [64, 96, 128, 16, 32, 32], use_bn=True)
        x = self.inception_module(x, [128, 128, 192, 32, 96, 64], use_bn=True, downsampler=True)

        # Stage 2
        x = self.inception_module(x, [192, 96, 208, 16, 48, 64], use_bn=True)
        x = self.inception_module(x, [160, 112, 224, 24, 64, 64], use_bn=True)
        x = self.inception_module(x, [128, 128, 256, 24, 64, 64], use_bn=True)
        x = self.inception_module(x, [112, 144, 288, 32, 64, 64], use_bn=True)
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128], use_bn=True, downsampler=True)

        # Stage 3
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128], use_bn=True)
        x = self.inception_module(x, [384, 192, 384, 48, 128, 128], use_bn=True)

        # Output
        output = AveragePooling2D((7, 7), strides=1)(x)
        output = Flatten()(output)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(output)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Inception_v1_BN_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model
