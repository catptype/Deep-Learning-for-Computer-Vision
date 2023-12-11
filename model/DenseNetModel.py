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
    Custom model implementing a DenseNet architecture.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        num_class (int): Number of classes for classification.
        growth_rate (int): Growth rate controlling the number of filters in the DenseNet model.
    
    Methods:
        init_block(input): Build the initial block of the DenseNet model.
        BN_ReLU_Conv(input, num_feature, kernel=3): Build a block with Batch Normalization, ReLU activation, and 3x3 Convolution.        
        Dense_block(input, num_feature, num_layer): Build a dense block in the DenseNet model.
        Transit_block(input): Build a transition block in the DenseNet model.

    Subclasses:
        - DenseNet121
        - DenseNet169
        - DenseNet201
        - DenseNet264
    """
    def __init__(self, image_size, num_class, growth_rate):
        self.image_size = image_size
        self.num_class = num_class
        self.growth_rate = growth_rate
        super().__init__()

    def init_block(self, input):
        x = Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        return x

    def BN_ReLU_Conv(self, input, num_feature, kernel=3):
        x = BatchNormalization()(input)
        x = ReLU()(x)
        x = Conv2D(num_feature, (kernel, kernel), padding="same", kernel_initializer="he_normal")(x)
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
    Subclass of DenseNetModel with specific configuration.

    Inherits from DenseNetModel.

    Example:
        ```python
        # Example usage to create a DenseNet121 model
        model = DenseNet121(image_size=224, num_class=10, growth_rate=32)
        ```

    Note: Inherits parameters from DenseNetModel.
    """
    def __init__(self, image_size, num_class, growth_rate=32):
        super().__init__(
            image_size=image_size, 
            num_class=num_class, 
            growth_rate=growth_rate,
        )

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
        x = self.init_block(input)

        # Dense block
        for idx, config in enumerate([6, 12, 24, 6]):
            x = self.Dense_block(x, self.growth_rate, config)
            if idx != 3: # ignore transit block at final
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
    Subclass of DenseNetModel with specific configuration.

    Inherits from DenseNetModel.

    Example:
        ```python
        # Example usage to create a DenseNet169 model
        model = DenseNet169(image_size=224, num_class=10, growth_rate=32)
        ```

    Note: Inherits parameters from DenseNetModel.
    """
    def __init__(self, image_size, num_class, growth_rate=32):
        super().__init__(
            image_size=image_size, 
            num_class=num_class, 
            growth_rate=growth_rate,
        )

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
    Subclass of DenseNetModel with specific configuration.

    Inherits from DenseNetModel.

    Example:
        ```python
        # Example usage to create a DenseNet201 model
        model = DenseNet201(image_size=224, num_class=10, growth_rate=32)
        ```

    Note: Inherits parameters from DenseNetModel.
    """ 
    def __init__(self, image_size, num_class, growth_rate=32):
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
    Subclass of DenseNetModel with specific configuration.

    Inherits from DenseNetModel.

    Example:
        ```python
        # Example usage to create a DenseNet264 model
        model = DenseNet264(image_size=224, num_class=10, growth_rate=32)
        ```

    Note: Inherits parameters from DenseNetModel.
    """
    def __init__(self, image_size, num_class, growth_rate=32):
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
