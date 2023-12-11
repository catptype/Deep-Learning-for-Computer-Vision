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
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel
from .SE_Module import SE_Module
from .CBAM_Module import CBAM_Module


class Res2NetModel(DeepLearningModel):
    """
    Custom model implementing the Res2Net architecture.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        num_class (int): Number of classes for classification.
        scale (int): Scale factor for the Res2Net architecture.
        module (str): Module type to enhance Res2Net blocks. Choose from ['se', 'cbam'].

    Methods:
        Conv2D_block(input, num_feature, kernel=3, strides=1, use_skip=False, identity=None): Build a block with Conv2D, Batch Normalization, ReLU activation, and optional downsampling.
        Residual_bottleneck(input, num_feature): Build a Residual Bottleneck block in the Res2Net model.

    Subclasses:
        - Res2Net50
        - Res2Net101
        - Res2Net152
    """
    def __init__(self, image_size, num_class, scale, module):
        if module is not None and module not in ['se', 'cbam']:
            raise ValueError(f"module value must be one of: 'None', 'se', 'cbam'")
    
        self.image_size = image_size
        self.num_class = num_class
        self.scale = scale
        self.module = module
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

        if self.module is None:
            x = self.Conv2D_block(x, num_feature, kernel=1, use_skip=True, identity=input)

        elif self.module == 'se':
            x = self.Conv2D_block(x, num_feature, kernel=1)
            x = SE_Module()(input=x, identity=input)

        elif self.module == 'cbam':
            x = self.Conv2D_block(x, num_feature, kernel=1)
            identity = x
            x = CBAM_Module(ratio=16)(x)
            x = Add()([x, identity])

        return x


class Res2Net50(Res2NetModel):
    """
    Subclass of Res2NetModel with specific configuration.

    Inherits from Res2NetModel.

    Example:
        ```python
        # Example usage to create a Res2Net50 model
        model = Res2Net50(image_size=224, num_class=10, scale=4, module='se')
        ```
    """
    def __init__(self, image_size, num_class, scale=4, module=None):
        super().__init__(
            image_size=image_size, 
            num_class=num_class, 
            scale=scale, 
            module=module,
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
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        if self.module is None:
            model_name = f"Res2Net50_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'se':
            model_name = f"Res2Net50SE_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'cbam':
            model_name = f"Res2Net50CBAM_{self.image_size}x{self.image_size}_{self.num_class}Class"

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model


class Res2Net101(Res2NetModel):
    """
    Subclass of Res2NetModel with specific configuration.

    Inherits from Res2NetModel.

    Example:
        ```python
        # Example usage to create a Res2Net101 model
        model = Res2Net101(image_size=224, num_class=10, scale=4, module='se')
        ```
    """
    def __init__(self, image_size, num_class, scale=4, module=None):
        super().__init__(
            image_size=image_size, 
            num_class=num_class, 
            scale=scale, 
            module=module,
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
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        if self.module is None:
            model_name = f"Res2Net101_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'se':
            model_name = f"Res2Net101SE_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'cbam':
            model_name = f"Res2Net101CBAM_{self.image_size}x{self.image_size}_{self.num_class}Class"

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model


class Res2Net152(Res2NetModel):
    """
    Subclass of Res2NetModel with specific configuration.

    Inherits from Res2NetModel.

    Example:
        ```python
        # Example usage to create a Res2Net152 model
        model = Res2Net152(image_size=224, num_class=10, scale=4, module='se')
        ```
    """
    def __init__(self, image_size, num_class, scale=4, module=None):
        super().__init__(
            image_size=image_size, 
            num_class=num_class, 
            scale=scale, 
            module=module,
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
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        if self.module is None:
            model_name = f"Res2Net152_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'se':
            model_name = f"Res2Net152SE_{self.image_size}x{self.image_size}_{self.num_class}Class"
        elif self.module == 'cbam':
            model_name = f"Res2Net152CBAM_{self.image_size}x{self.image_size}_{self.num_class}Class"

        model = Model(inputs=[input], outputs=output, name=model_name)
        return model
