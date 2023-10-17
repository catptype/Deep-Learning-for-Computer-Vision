import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel


class VGGModel(DeepLearningModel):
    """
    Base class for VGG architectures.

    This class serves as the base for implementing various VGG architectures.

    Parameters:
        image_size (int): The input image size.
        num_class (int): The number of output classes for classification.

    Methods:
        Conv2D_block(input, num_feature, kernel=3, use_bn=False, downsampler=False):
            Apply a VGG-style convolutional block with optional batch normalization and downsampling.
    """
    def __init__(self, image_size, num_class):
        self.image_size = image_size
        self.num_class = num_class
        super().__init__()

    def Conv2D_block(self, input, num_feature, kernel=3, use_bn=False, downsampler=False):
        """
        Create a VGG-style convolutional block with optional batch normalization and downsampling.

        Parameters:
            input: Input tensor for the convolutional block.
            num_feature (int): The number of output feature maps.
            kernel (int, optional): The kernel size for convolution. Default is 3.
            use_bn (bool, optional): Whether to use batch normalization. Default is False.
            downsampler (bool, optional): Whether to include downsampling layers. Default is False.

        Returns:
            TensorFlow tensor representing the output of the convolutional block.
        """
        x = Conv2D(num_feature, (kernel, kernel), padding="same", kernel_initializer="he_normal")(input)
        if use_bn:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        if downsampler:
            x = MaxPooling2D(strides=2)(x)
        return x


class CustomVGG(VGGModel):
    """
    Implementation of a custom VGG architecture.
    """
    def __init__(self, image_size, num_class):
        """
        Initializes the CustomVGG model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_class=num_class)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64, use_bn=True)
        x = self.Conv2D_block(x, 64, use_bn=True, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128, use_bn=True)
        x = self.Conv2D_block(x, 128, use_bn=True, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256, use_bn=True)
        x = self.Conv2D_block(x, 256, use_bn=True, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"CustomVGG_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class VGG11(VGGModel):
    """
    Implementation of the VGG11 architecture.
    """
    def __init__(self, image_size, num_class):
        """
        Initializes the VGG11 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_class=num_class)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG11_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class VGG13(VGGModel):
    """
    Implementation of the VGG13 architecture.
    """
    def __init__(self, image_size, num_class):
        """
        Initializes the VGG13 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_class=num_class)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(x, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128)
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG13_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class VGG16(VGGModel):
    """
    Implementation of the VGG16 architecture.
    """
    def __init__(self, image_size, num_class):
        """
        Initializes the VGG16 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_class=num_class)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(x, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128)
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG16_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class VGG19(VGGModel):
    """
    Implementation of the VGG19 architecture.
    """
    def __init__(self, image_size, num_class):
        """
        Initializes the VGG19 model with specified parameters.
        
        Args:
            image_size (int): The input image size.
            num_class (int): The number of output classes.
        """
        super().__init__(image_size=image_size, num_class=num_class)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(input, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(input, 128)
        x = self.Conv2D_block(input, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG19_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model
