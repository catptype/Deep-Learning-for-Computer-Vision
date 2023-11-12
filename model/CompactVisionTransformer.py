from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
)
import tensorflow as tf
from .transformer_module.ImagePatcher import ImagePatcher
from .transformer_module.PatchEncoder import PatchEncoder
from .transformer_module.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel

class CompactVisionTransformer(DeepLearningModel):
    """
    Compact Vision Transformers (CVT).

    This class defines a custom deep learning model for Compact Vision Transformers (CVT)
    that consists of an image patcher, patch encoder, transformer encoder, sequence pooling, and a classification head.

    Attributes:
        image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
        patch_size (int): The size of image patches.
        num_class (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the transformer.
        latent_size (int): The latent dimension size for patch embeddings.
        num_transformer (int): The number of transformer encoder layers.
        mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.

    """
    def __init__(
        self,
        image_size,
        patch_size,
        num_class,
        num_head,
        latent_size,
        num_transformer,
        mlp_size,
    ):
        """
        Constructor method for initializing the CVTModel.

        Args:
            image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
            patch_size (int): The size of image patches.
            num_class (int): The number of output classes for classification.
            num_head (int): The number of attention heads in the transformer.
            latent_size (int): The latent dimension size for patch embeddings.
            num_transformer (int): The number of transformer encoder layers.
            mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
        """
        if image_size % patch_size != 0:
            raise ValueError(f"Image size ({image_size}) is not divisible by the patch size ({patch_size})")

        if latent_size % num_head != 0:
            raise ValueError(f"Latent size ({latent_size}) is not divisible by the number of attention heads ({num_head})")

        self.image_size = image_size
        self.num_class = num_class
        self.patch_size = patch_size
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_transformer = num_transformer
        self.mlp_size = mlp_size
        super().__init__()

    def build_model(self):
        """
        Method for building the CVTModel architecture and returning the Keras model.

        Returns:
            model (Model): The Keras model representing the CVTModel.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Image patcher
        x = ImagePatcher(patch_size=self.patch_size)(input)
        num_patch = x.shape[1]

        # Patch encoder
        x = PatchEncoder(num_patch=num_patch, latent_size=self.latent_size)(x)

        # Transformer encoder
        for _ in range(self.num_transformer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # SeqPool
        SeqPool = Dense(1, activation="softmax")(x)        
        SeqPool = tf.matmul(SeqPool, x, transpose_a=True)

        # Classification
        x = Flatten()(SeqPool)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x)

        model_name = f"CVT_I{self.image_size}x{self.image_size}_P{self.patch_size}_L{self.num_transformer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_class}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model