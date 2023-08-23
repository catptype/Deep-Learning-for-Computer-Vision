"""
Res2Net Models
~~~~~~~~~~~~~

This module defines a set of Res2Net architectures for image classification tasks using TensorFlow and Keras.

Classes:
    - Res2NetModel: Base class for Res2Net architectures.
    - Res2Net50: Implementation of Res2Net-50 architecture.
    - Res2Net101: Implementation of Res2Net-101 architecture.
    - Res2Net152: Implementation of Res2Net-152 architecture.

Note: To use these models, TensorFlow and Keras must be installed.
"""
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Multiply,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel
from .SE_Module import SE_Module


class Res2NetModel(DeepLearningModel):
    """
    Base class for Res2Net architectures.
    
    Attributes:
        - scale: The scale factor for the Res2Net model.
        - use_se: Whether to use Squeeze-and-Excitation (SE) blocks.
    
    Methods:
        - Conv2D_block(input, num_feature, kernel, strides, use_skip, identity): Creates a Convolutional Block.
        - SE_block(input, identity, ratio): Creates a Squeeze-and-Excitation (SE) block.
        - Residual_bottleneck(input, num_feature): Creates a Residual Bottleneck block.
    """
    def __init__(self, image_size, num_classes, **kwargs):
        """
        Initializes the Res2Net model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - **kwargs: Additional keyword arguments.
        """
        self.scale = kwargs["scale"]
        self.use_se = kwargs["use_se"]
        super().__init__(image_size=image_size, num_classes=num_classes)

    def Conv2D_block(self, input, num_feature, kernel=3, strides=1, use_skip=False, identity=None):
        x = Conv2D(num_feature, (kernel, kernel), strides=strides, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        if use_skip:
            if x.shape[-1] != identity.shape[-1]:
                identity = self.Conv2D_block(identity, x.shape[-1], kernel=1)
            x = Add()([x, identity])
        x = ReLU()(x)
        return x
    
    def Residual_bottleneck(self, input, num_feature):
        module_output = []

        x = self.Conv2D_block(input, num_feature)

        tensor_split = tf.split(x, num_or_size_splits=self.scale, axis=-1)  # Output as list

        for idx, tensor in enumerate(tensor_split):
            if idx == 0:
                module_output.append(tensor)
            elif idx == 1:
                k = self.Conv2D_block(tensor, num_feature // self.scale)
                module_output.append(k)
            else:
                k_identity = module_output[idx - 1]
                k = Add()([tensor, k_identity])
                k = self.Conv2D_block(k, num_feature // self.scale)
                module_output.append(k)

        x = Concatenate()(module_output)

        if self.use_se:
            x = self.Conv2D_block(x, num_feature, kernel=1)
            with tf.name_scope("SE_Module"):
                x = SE_Module()(input=x, identity=input)
        else:
            x = self.Conv2D_block(x, num_feature, kernel=1, use_skip=True, identity=input)

        return x


class Res2Net50(Res2NetModel):
    """
    Implementation of Res2Net-50 architecture.
    """
    def __init__(self, image_size, num_classes, scale=4, use_se=False):
        """
        Initializes the Res2Net-50 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - scale (int): The scale factor for the model.
            - use_se (bool): Whether to use Squeeze-and-Excitation (SE) blocks.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, scale=scale, use_se=use_se)

    def build_model(self):
        """
        Builds the Res2Net-50 model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for _ in range(4):
            x = self.Residual_bottleneck(x, 128)

        # stage 3
        for _ in range(6):
            x = self.Residual_bottleneck(x, 256)

        # stage 4
        for _ in range(3):
            x = self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        if self.use_se:
            model_name = f"Res2Net50SE_{self.image_size}x{self.image_size}_{self.num_classes}Class"
        else:
            model_name = (f"Res2Net50_{self.image_size}x{self.image_size}_{self.num_classes}Class")

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model


class Res2Net101(Res2NetModel):
    """
    Implementation of Res2Net-101 architecture.
    """
    def __init__(self, image_size, num_classes, scale=4, use_se=False):
        """
        Initializes the Res2Net-101 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - scale (int): The scale factor for the model.
            - use_se (bool): Whether to use Squeeze-and-Excitation (SE) blocks.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, scale=scale, use_se=use_se)

    def build_model(self):
        """
        Builds the Res2Net-101 model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for _ in range(4):
            x = self.Residual_bottleneck(x, 128)

        # stage 3
        for _ in range(23):
            x = self.Residual_bottleneck(x, 256)

        # stage 4
        for _ in range(3):
            x = self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        if self.use_se:
            model_name = f"Res2Net101SE_{self.image_size}x{self.image_size}_{self.num_classes}Class"
        else:
            model_name = f"Res2Net101_{self.image_size}x{self.image_size}_{self.num_classes}Class"

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model


class Res2Net152(Res2NetModel):
    """
    Implementation of Res2Net-152 architecture.
    """
    def __init__(self, image_size, num_classes, scale=4, use_se=False):
        """
        Initializes the Res2Net-152 model with specified parameters.
        
        Args:
            - image_size (int): The input image size.
            - num_classes (int): The number of output classes.
            - scale (int): The scale factor for the model.
            - use_se (bool): Whether to use Squeeze-and-Excitation (SE) blocks.
        """
        super().__init__(image_size=image_size, num_classes=num_classes, scale=scale, use_se=use_se)

    def build_model(self):
        """
        Builds the Res2Net-152 model.
        
        Returns:
            - model: The built Keras model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for _ in range(8):
            x = self.Residual_bottleneck(x, 128)

        # stage 3
        for _ in range(36):
            x = self.Residual_bottleneck(x, 256)

        # stage 4
        for _ in range(3):
            x = self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        if self.use_se:
            model_name = f"Res2Net152SE_{self.image_size}x{self.image_size}_{self.num_classes}Class"
        else:
            model_name = f"Res2Net152_{self.image_size}x{self.image_size}_{self.num_classes}Class"

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model
