import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel


class ResnetModel(DeepLearningModel):
    """
    Custom model implementing the ResNet architecture.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        num_class (int): Number of classes for classification.

    Methods:
        Conv2D_block(input, num_feature, kernel=3, strides=1, use_skip=False, identity=None): Build a block with Conv2D, Batch Normalization, ReLU activation, and optional downsampling.
        Residual_block(input, num_feature, downsampler=False): Build a Residual block in the ResNet model.
        Residual_bottleneck(input, num_feature, downsampler=False): Build a Residual Bottleneck block in the ResNet model.

    Subclasses:
        - Resnet18
        - Resnet34
        - Resnet50
        - Resnet101
        - Resnet152
    """
    def __init__(self, image_size, num_class):
        self.image_size = image_size
        self.num_class = num_class
        super().__init__()

    def Conv2D_block(self, input, num_feature, kernel=3, strides=1, use_skip=False, identity=None):
        x = Conv2D(num_feature, (kernel, kernel), strides=strides, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        if use_skip:
            if x.shape[-1] != identity.shape[-1]:
                identity = self.Conv2D_block(identity, x.shape[-1], kernel=1)
            x = Add()([x, identity])
        x = ReLU()(x)
        return x

    def Residual_block(self, input, num_feature, downsampler=False):
        if downsampler:
            x = self.Conv2D_block(input, num_feature, strides=2)
            x = self.Conv2D_block(x, num_feature, use_skip=True, identity=MaxPooling2D()(input))
        else:
            x = self.Conv2D_block(input, num_feature)
            x = self.Conv2D_block(x, num_feature, use_skip=True, identity=input)
        return x

    def Residual_bottleneck(self, input, num_feature, downsampler=False):
        x = self.Conv2D_block(input, num_feature, kernel=1)
        if downsampler:
            x = self.Conv2D_block(x, num_feature, strides=2)
            x = self.Conv2D_block(x, num_feature * 4, kernel=1, use_skip=True, identity=MaxPooling2D()(input),)
        else:
            x = self.Conv2D_block(x, num_feature)
            x = self.Conv2D_block(x, num_feature * 4, kernel=1, use_skip=True, identity=input)
        return x


class Resnet18(ResnetModel):
    """
    Subclass of ResnetModel with specific configuration.

    Inherits from ResnetModel.

    Example:
        ```python
        # Example usage to create a ResNet18 model
        model = Resnet18(image_size=224, num_class=10)
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

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        x = self.Residual_block(x, 64)
        x = self.Residual_block(x, 64)

        # stage 2
        x = self.Residual_block(x, 128, downsampler=True)
        x = self.Residual_block(x, 128)

        # stage 3
        x = self.Residual_block(x, 256, downsampler=True)
        x = self.Residual_block(x, 256)

        # stage 4
        x = self.Residual_block(x, 512, downsampler=True)
        x = self.Residual_block(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Resnet18_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class Resnet34(ResnetModel):
    """
    Subclass of ResnetModel with specific configuration.

    Inherits from ResnetModel.

    Example:
        ```python
        # Example usage to create a ResNet34 model
        model = Resnet34(image_size=224, num_class=10)
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

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_block(x, 64)

        # stage 2
        for i in range(4):
            x = self.Residual_block(x, 128, downsampler=True) if i == 0 else self.Residual_block(x, 128)

        # stage 3
        for i in range(6):
            x = self.Residual_block(x, 256, downsampler=True) if i == 0 else self.Residual_block(x, 256)

        # stage 4
        for i in range(3):
            x = self.Residual_block(x, 512, downsampler=True) if i == 0 else self.Residual_block(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Resnet34_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class Resnet50(ResnetModel):
    """
    Subclass of ResnetModel with specific configuration.

    Inherits from ResnetModel.

    Example:
        ```python
        # Example usage to create a ResNet50 model
        model = Resnet50(image_size=224, num_class=10)
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

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for i in range(4):
            x = self.Residual_bottleneck(x, 128, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 128)

        # stage 3
        for i in range(6):
            x = self.Residual_bottleneck(x, 256, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 256)

        # stage 4
        for i in range(3):
            x = self.Residual_bottleneck(x, 512, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Resnet50_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class Resnet101(ResnetModel):
    """
    Subclass of ResnetModel with specific configuration.

    Inherits from ResnetModel.

    Example:
        ```python
        # Example usage to create a ResNet101 model
        model = Resnet101(image_size=224, num_class=10)
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

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for i in range(4):
            x = self.Residual_bottleneck(x, 128, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 128)

        # stage 3
        for i in range(23):
            x = self.Residual_bottleneck(x, 256, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 256)

        # stage 4
        for i in range(3):
            x = self.Residual_bottleneck(x, 512, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Resnet101_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model


class Resnet152(ResnetModel):
    """
    Subclass of ResnetModel with specific configuration.

    Inherits from ResnetModel.

    Example:
        ```python
        # Example usage to create a ResNet152 model
        model = Resnet152(image_size=224, num_class=10)
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

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Residual_bottleneck(x, 64)

        # stage 2
        for i in range(8):
            x = self.Residual_bottleneck(x, 128, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 128)

        # stage 3
        for i in range(36):
            x = self.Residual_bottleneck(x, 256, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 256)

        # stage 4
        for i in range(3):
            x = self.Residual_bottleneck(x, 512, downsampler=True) if i == 0 else self.Residual_bottleneck(x, 512)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"Resnet152_{self.image_size}x{self.image_size}_{self.num_class}Class",
        )
        return model