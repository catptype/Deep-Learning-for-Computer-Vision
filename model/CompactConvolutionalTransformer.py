import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input

from .transformer_module.ConvToken import ConvToken
from .transformer_module.PatchEncoder import PatchEncoder
from .transformer_module.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel

class CompactConvolutionalTransformer(DeepLearningModel):
    """
    Custom model implementing a Compact Convolutional Transformer.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        conv_layer (list): List of integers specifying the number of filters for each convolutional layer.
        num_class (int): Number of classes for classification.
        num_head (int): Number of attention heads in the Transformer encoder.
        latent_size (int): Dimensionality of the latent space in the Transformer encoder.
        num_transformer (int): Number of Transformer encoder blocks.
        mlp_size (int): Size of the feedforward layer in the Transformer encoder.
        position_embedding (bool): Whether to use positional embeddings in the PatchEncoder.
    
    Methods:
        build_model(): Build the Compact Convolutional Transformer model.

    Example:
        ```python
        # Example usage to create a CompactConvolutionalTransformer model
        model = CompactConvolutionalTransformer(
            image_size=224,
            conv_layer=[64, 128, 256],
            num_class=10,
            num_head=4,
            latent_size=256,
            num_transformer=6,
            mlp_size=512,
            position_embedding=True,
        )
        ```

    Subclasses:
        - CCT2
        - CCT4
        - CCT6
        - CCT7
        - CCT14
    """
    def __init__(
        self,
        image_size,
        conv_layer,
        num_class,
        num_head,
        latent_size,
        num_transformer,
        mlp_size,
        position_embedding=False,
    ):
        if latent_size % num_head != 0:
            raise ValueError(f"Latent size ({latent_size}) is not divisible by the number of attention heads ({num_head})")

        if latent_size != conv_layer[-1]:
            raise ValueError(f"Latent size ({latent_size}) must equal to Convolutional Tokenizer channel ({conv_layer[-1]})")

        self.image_size = image_size
        self.num_class = num_class
        self.conv_layer = conv_layer
        self.position_embedding = position_embedding
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_transformer = num_transformer
        self.mlp_size = mlp_size
        super().__init__()

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Convolution Token
        x = ConvToken(conv_layer=self.conv_layer)(input)
        
        # Patch encoder
        if self.position_embedding:
            x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Transformer encoder
        for _ in range(self.num_transformer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # SeqPool
        SeqPool = Dense(1, activation="softmax")(x)        
        SeqPool = tf.matmul(SeqPool, x, transpose_a=True)

        # Classification
        x = Flatten()(SeqPool)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        conv_name = ""
        for idx, layer in enumerate(self.conv_layer):
            if idx == 0:
                conv_name += str(layer)
            else:
                conv_name += f"_{layer}"

        model_name = f"CCT_I{self.image_size}x{self.image_size}_Conv{conv_name}_L{self.num_transformer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_class}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model
    
class CCT2(CompactConvolutionalTransformer):
    """
    Subclass of CompactConvolutionalTransformer with specific configuration.

    Inherits from CompactConvolutionalTransformer.

    Example:
        ```python
        # Example usage to create a CCT2 model
        model = CCT2(image_size=224, conv_layer=[64, 128, 256], num_class=10)
        ```

    Note: Inherits parameters from CompactConvolutionalTransformer.
    """
    def __init__(self, image_size, conv_layer, num_class):
        super().__init__(
            image_size=image_size,
            conv_layer=conv_layer,
            num_class=num_class,
            num_head=2,
            latent_size=128,
            num_transformer=2,
            mlp_size=128,
            position_embedding=False,
        )

class CCT4(CompactConvolutionalTransformer):
    """
    Subclass of CompactConvolutionalTransformer with specific configuration.

    Inherits from CompactConvolutionalTransformer.

    Example:
        ```python
        # Example usage to create a CCT4 model
        model = CCT4(image_size=224, conv_layer=[64, 128, 256], num_class=10)
        ```

    Note: Inherits parameters from CompactConvolutionalTransformer.
    """
    def __init__(self, image_size, conv_layer, num_class):
        super().__init__(
            image_size=image_size,
            conv_layer=conv_layer,
            num_class=num_class,
            num_head=2,
            latent_size=128,
            num_transformer=4,
            mlp_size=128,
            position_embedding=False,
        )

class CCT6(CompactConvolutionalTransformer):
    """
    Subclass of CompactConvolutionalTransformer with specific configuration.

    Inherits from CompactConvolutionalTransformer.

    Example:
        ```python
        # Example usage to create a CCT6 model
        model = CCT6(image_size=224, conv_layer=[64, 128, 256], num_class=10)
        ```

    Note: Inherits parameters from CompactConvolutionalTransformer.
    """
    def __init__(self, image_size, conv_layer, num_class):
        super().__init__(
            image_size=image_size,
            conv_layer=conv_layer,
            num_class=num_class,
            num_head=4,
            latent_size=256,
            num_transformer=6,
            mlp_size=512,
            position_embedding=False,
        )

class CCT7(CompactConvolutionalTransformer):
    """
    Subclass of CompactConvolutionalTransformer with specific configuration.

    Inherits from CompactConvolutionalTransformer.

    Example:
        ```python
        # Example usage to create a CCT7 model
        model = CCT7(image_size=224, conv_layer=[64, 128, 256], num_class=10)
        ```

    Note: Inherits parameters from CompactConvolutionalTransformer.
    """
    def __init__(self, image_size, conv_layer, num_class):
        super().__init__(
            image_size=image_size,
            conv_layer=conv_layer,
            num_class=num_class,
            num_head=4,
            latent_size=256,
            num_transformer=7,
            mlp_size=512,
            position_embedding=False,
        )

class CCT14(CompactConvolutionalTransformer):
    """
    Subclass of CompactConvolutionalTransformer with specific configuration.

    Inherits from CompactConvolutionalTransformer.

    Example:
        ```python
        # Example usage to create a CCT14 model
        model = CCT14(image_size=224, conv_layer=[64, 128, 256], num_class=10)
        ```

    Note: Inherits parameters from CompactConvolutionalTransformer.
    """
    def __init__(self, image_size, conv_layer, num_class):
        super().__init__(
            image_size=image_size,
            conv_layer=conv_layer,
            num_class=num_class,
            num_head=6,
            latent_size=384,
            num_transformer=14,
            mlp_size=1152,
            position_embedding=False,
        )