import sys
sys.dont_write_bytecode = True

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
)
import tensorflow as tf
from .transformer_module.ConvToken import ConvToken
from .transformer_module.PatchEncoder import PatchEncoder
from .transformer_module.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel

class CompactConvolutionalTransformer(DeepLearningModel):
    """
    Compact Convolutional Transformer (CCT).

    This class defines a custom deep learning model for CCT
    that combines convolutional tokenization, position embeddings, transformer encoding, sequence pooling,
    and a classification head.

    Attributes:
        image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
        conv_layers (list): A list of integers representing the number of features for each convolutional layer.
        num_classes (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the transformer.
        latent_size (int): The latent dimension size for patch embeddings.
        num_layer (int): The number of transformer encoder layers.
        mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
        position_embedding (bool): A flag indicating whether to include position embeddings.

    """
    def __init__(
        self,
        image_size,
        conv_layer,
        num_classes,
        num_head,
        latent_size,
        num_layer,
        mlp_size,
        position_embedding=False,
    ):
        """
        Constructor method for initializing the CCT.

        Args:
            image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
            conv_layer (list): A list of integers representing the number of features
                               for each convolutional layer.
            num_classes (int): The number of output classes for classification.
            num_head (int): The number of attention heads in the transformer.
            latent_size (int): The latent dimension size for patch embeddings.
            num_layer (int): The number of transformer encoder layers.
            mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
            position_embedding (bool): A flag indicating whether to include position embeddings.
        """
        assert latent_size % num_head == 0, f"Latent size ({latent_size}) is not divisible by number of attention heads ({num_head})"

        self.conv_layer = conv_layer
        self.position_embedding = position_embedding
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_layer = num_layer
        self.mlp_size = mlp_size
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Method for building the CCT architecture and returning the Keras model.

        Returns:
            model (Model): The Keras model representing the CCTModel.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Convolution Token
        x = ConvToken(conv_layer=self.conv_layer)(input)
        assert self.latent_size == x.shape[-1], f"Convolutional Tokenizer must has channel ({x.shape[-1]}) = embedding dimension ({self.latent_size})"
        
        # Patch encoder
        if self.position_embedding:
            x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Transformer encoder
        for _ in range(self.num_layer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # SeqPool
        SeqPool = Dense(1, activation="softmax")(x)        
        SeqPool = tf.matmul(SeqPool, x, transpose_a=True)

        # Classification
        x = Flatten()(SeqPool)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model_name = f"CCT_I{self.image_size}x{self.image_size}_Conv{self.conv_layer}_L{self.num_layer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_classes}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model