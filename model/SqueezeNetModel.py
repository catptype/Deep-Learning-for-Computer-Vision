"""
SqueezeNet Models
~~~~~~~~~~~~~~~~~

This module defines a set of SqueezeNet architectures for image classification tasks using TensorFlow and Keras.

Classes:
    - SqueezeNetModel: Base class for SqueezeNet architectures.
    - SqueezeNet: Implementation of the SqueezeNet architecture.
    - SqueezeNet_BN: Implementation of the SqueezeNet architecture with Batch Normalization.
    - SqueezeNet_SimpleSkip: Implementation of the SqueezeNet architecture with simple skip connections.
    - SqueezeNet_SimpleSkip_BN: Implementation of the SqueezeNet architecture with simple skip connections and Batch Normalization.
    - SqueezeNet_ComplexSkip: Implementation of the SqueezeNet architecture with complex skip connections.
    - SqueezeNet_ComplexSkip_BN: Implementation of the SqueezeNet architecture with complex skip connections and Batch Normalization.

Note: TensorFlow and Keras must be installed to use these models.]
"""
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
    Softmax,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel


class SqueezeNetModel(DeepLearningModel):
    """
    Base class for SqueezeNet architectures.
    
    Methods:
        - Conv2D_block(input, num_feature, kernel, strides, use_bn): Creates a Convolutional Block.
        - init_block(input, use_bn): Creates the initial block of the architecture.
        - fire_module(input, s1x1_feature, e1x1_feature, e3x3_feature, downsampler, use_bn, use_skip, identity): Creates a Fire module.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet model with specified parameters.
        
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
        x = self.Conv2D_block(input, 96, kernel=7, strides=2, use_bn=use_bn)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        return x

    def fire_module(
        self,
        input,
        s1x1_feature,
        e1x1_feature,
        e3x3_feature,
        downsampler=False,
        use_bn=False,
        use_skip=False,
        identity=None,
    ):
        assert (s1x1_feature < e1x1_feature + e3x3_feature), f"s1x1({s1x1_feature}) must less than sum of e1x1 and e3x3({e1x1_feature+e3x3_feature})"

        # Squeeze 1x1
        squeeze = self.Conv2D_block(input, s1x1_feature, kernel=1, use_bn=use_bn)

        # Expand 1x1 and 3x3
        expand1x1 = self.Conv2D_block(squeeze, e1x1_feature, kernel=1, use_bn=use_bn)
        expand3x3 = self.Conv2D_block(squeeze, e3x3_feature, use_bn=use_bn)

        output = Concatenate()([expand1x1, expand3x3])
        if use_skip:
            if output.shape[-1] != identity.shape[-1]:
                identity = self.Conv2D_block(identity, output.shape[-1], kernel=1, use_bn=use_bn)
            output = Add()([output, identity])

        if downsampler:
            output = MaxPooling2D((3, 3), strides=2)(output)
        return output

class SqueezeNet(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64)
        x = self.fire_module(x, 16, 64, 64)
        x = self.fire_module(x, 32, 128, 128, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128)
        x = self.fire_module(x, 48, 192, 192)
        x = self.fire_module(x, 48, 192, 192)
        x = self.fire_module(x, 64, 256, 256, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model
    

class SqueezeNet_BN(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture with Batch Normalization.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet_BN model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet_BN model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input, use_bn=True)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64, use_bn=True)
        x = self.fire_module(x, 16, 64, 64, use_bn=True)
        x = self.fire_module(x, 32, 128, 128, use_bn=True, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128, use_bn=True)
        x = self.fire_module(x, 48, 192, 192, use_bn=True)
        x = self.fire_module(x, 48, 192, 192, use_bn=True)
        x = self.fire_module(x, 64, 256, 256, use_bn=True, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256, use_bn=True)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1, use_bn=True)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_BN_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class SqueezeNet_SimpleSkip(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture with simple skip connections.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet_SimpleSkip model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet_SimpleSkip model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64)
        x = self.fire_module(x, 16, 64, 64, use_skip=True, identity=x)
        x = self.fire_module(x, 32, 128, 128, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192)
        x = self.fire_module(x, 48, 192, 192, use_skip=True, identity=x)
        x = self.fire_module(x, 64, 256, 256, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256, use_skip=True, identity=x)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_SimpleSkip_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model
    
class SqueezeNet_SimpleSkip_BN(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture with simple skip connections and Batch Normalization.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet_SimpleSkip_BN model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet_SimpleSkip_BN model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input, use_bn=True)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64, use_bn=True)
        x = self.fire_module(x, 16, 64, 64, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 32, 128, 128, use_bn=True, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192, use_bn=True)
        x = self.fire_module(x, 48, 192, 192, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 64, 256, 256, use_bn=True, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256, use_bn=True, use_skip=True, identity=x)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1, use_bn=True)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_SimpleSkip_BN_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model


class SqueezeNet_ComplexSkip(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture with complex skip connections.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet_ComplexSkip model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet_ComplexSkip model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64, use_skip=True, identity=x)
        x = self.fire_module(x, 16, 64, 64, use_skip=True, identity=x)
        x = self.fire_module(x, 32, 128, 128, use_skip=True, identity=x, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192, use_skip=True, identity=x)
        x = self.fire_module(x, 64, 256, 256, use_skip=True, identity=x, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256, use_skip=True, identity=x)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_ComplexSkip_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model

class SqueezeNet_ComplexSkip_BN(SqueezeNetModel):
    """
    Implementation of the SqueezeNet architecture with complex skip connections and Batch Normalization.
    """
    def __init__(self, image_size, num_classes):
        """
        Initializes the SqueezeNet_ComplexSkip_BN model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the SqueezeNet_ComplexSkip_BN model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Initial block
        x = self.init_block(input, use_bn=True)

        # Stage 1
        x = self.fire_module(x, 16, 64, 64, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 16, 64, 64, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 32, 128, 128, use_bn=True, use_skip=True, identity=x, downsampler=True)

        # Stage 2
        x = self.fire_module(x, 32, 128, 128, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 48, 192, 192, use_bn=True, use_skip=True, identity=x)
        x = self.fire_module(x, 64, 256, 256, use_bn=True, use_skip=True, identity=x, downsampler=True)

        # Output
        x = self.fire_module(x, 64, 256, 256, use_skip=True, identity=x)
        x = Dropout(0.5)(x)
        x = self.Conv2D_block(x, self.num_classes, kernel=1, use_bn=True)
        x = GlobalAveragePooling2D()(x)
        output = Softmax(dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"SqueezeNet_ComplexSkip_BN_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )

        return model