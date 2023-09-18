import sys
sys.dont_write_bytecode = True

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    LayerNormalization,
)
import tensorflow as tf
from .transformer_module.ClassToken import ClassToken
from .transformer_module.ImagePatcher import ImagePatcher
from .transformer_module.PatchEncoder import PatchEncoder
from .transformer_module.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel


class VisionTransformer(DeepLearningModel):
    """
    Vision Transformer (ViT).

    This model consists of multiple TransformerEncoder blocks to process image patches and classify images.

    Args:
        image_size (int): The size of the input image (height and width) in pixels.
        patch_size (int): The size of each square image patch in pixels.
        num_classes (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the multi-head self-attention mechanism.
        latent_size (int): The size of the latent space for the encoder.
        num_layer (int): The number of TransformerEncoder layers in the model.
        mlp_size (int): The size of the feedforward neural network hidden layer.

    """
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        num_head,
        latent_size,
        num_layer,
        mlp_size,
    ):
        assert image_size % patch_size == 0, f"Image size ({image_size}) is not divisible by Patch size ({patch_size})"
        assert latent_size % num_head == 0, f"Latent size ({latent_size}) is not divisible by number of attention heads ({num_head})"

        self.patch_size = patch_size
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_layer = num_layer
        self.mlp_size = mlp_size
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the ViT model by stacking various layers.

        Returns:
            tf.keras.Model: The Vision Transformer model.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Image patcher
        x = ImagePatcher(patch_size=self.patch_size)(input)

        # Patch encoder
        x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Class Token
        x = ClassToken()(x)

        # Transformer encoder
        for _ in range(self.num_layer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # Classification
        x = LayerNormalization(epsilon=1e-6)(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x[:, 0, :])

        model_name = f"ViT_I{self.image_size}x{self.image_size}_P{self.patch_size}_L{self.num_layer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_classes}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model