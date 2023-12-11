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
    Custom model implementing the VGG architecture.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        num_class (int): Number of classes for classification.

    Methods:
        Conv2D_block(input, num_feature, kernel=3, use_bn=False, downsampler=False): Build a block with Conv2D, Batch Normalization, ReLU activation, and optional downsampling.

    Subclasses:
        - VGG11
        - VGG13
        - VGG16
        - VGG19
    """
    def __init__(self, image_size, num_class):
        self.image_size = image_size
        self.num_class = num_class
        super().__init__()

    def Conv2D_block(self, input, num_feature, kernel=3, use_bn=False, downsampler=False):
        x = Conv2D(num_feature, (kernel, kernel), padding="same", kernel_initializer="he_normal")(input)
        if use_bn:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        if downsampler:
            x = MaxPooling2D(strides=2)(x)
        return x

class VGG11(VGGModel):
    """
    Subclass of VGGModel with specific configuration.

    Inherits from VGGModel.

    Example:
        ```python
        # Example usage to create a VGG11 model
        model = VGG11(image_size=224, num_class=10)
        ```
    """
    def __init__(self, image_size, num_class):
        super().__init__(
            image_size=image_size, 
            num_class=num_class,
        )

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
    Subclass of VGGModel with specific configuration.

    Inherits from VGGModel.

    Example:
        ```python
        # Example usage to create a VGG13 model
        model = VGG13(image_size=224, num_class=10)
        ```
    """
    def __init__(self, image_size, num_class):
        super().__init__(
            image_size=image_size, 
            num_class=num_class,
        )

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
    Subclass of VGGModel with specific configuration.

    Inherits from VGGModel.

    Example:
        ```python
        # Example usage to create a VGG16 model
        model = VGG16(image_size=224, num_class=10)
        ```
    """
    def __init__(self, image_size, num_class):
        super().__init__(
            image_size=image_size, 
            num_class=num_class,
        )

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
    Subclass of VGGModel with specific configuration.

    Inherits from VGGModel.

    Example:
        ```python
        # Example usage to create a VGG19 model
        model = VGG19(image_size=224, num_class=10)
        ```
    """
    def __init__(self, image_size, num_class):
        super().__init__(
            image_size=image_size, 
            num_class=num_class,
        )

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
