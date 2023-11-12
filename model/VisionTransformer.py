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
        num_class (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the multi-head self-attention mechanism.
        latent_size (int): The size of the latent space for the encoder.
        num_transformer (int): The number of TransformerEncoder layers in the model.
        mlp_size (int): The size of the feedforward neural network hidden layer.

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
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Image patcher
        x = ImagePatcher(patch_size=self.patch_size)(input)

        # Patch encoder
        x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Class Token
        x = ClassToken()(x)

        # Transformer encoder
        for _ in range(self.num_transformer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # Classification
        x = LayerNormalization(epsilon=1e-6)(x)
        output = Dense(self.num_class, activation="softmax", dtype=tf.float32)(x[:, 0, :])

        model_name = f"ViT_I{self.image_size}x{self.image_size}_P{self.patch_size}_L{self.num_transformer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_class}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model
    

class ViTBase(VisionTransformer):
    """
    ViTBase is a variant of the Vision Transformer (ViT) architecture with customizable input size and number of classes.

    Args:
        image_size (int): The size of the input images.
        patch_size (int): The size of image patches for self-attention.
        num_class (int): The number of output classes for classification.

    Fixed Hyperparameters:
        num_head (int): Number of self-attention heads (Fixed to 12).
        latent_size (int): Dimensionality of latent space (Fixed to 768).
        num_transformer (int): Number of encoder layers (Fixed to 12).
        mlp_size (int): Size of the feedforward neural network in each layer (Fixed to 3072).

    """
    def __init__(self, image_size, patch_size, num_class):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_class=num_class,
            num_head=12,         
            latent_size=768,     
            num_transformer=12,        
            mlp_size=3072        
        )


class ViTLarge(VisionTransformer):
    """
    ViTLarge is a variant of the Vision Transformer (ViT) architecture with customizable input size and number of classes.

    Args:
        image_size (int): The size of the input images.
        patch_size (int): The size of image patches for self-attention.
        num_class (int): The number of output classes for classification.

    Fixed Hyperparameters:
        num_head (int): Number of self-attention heads (Fixed to 16).
        latent_size (int): Dimensionality of latent space (Fixed to 1024).
        num_transformer (int): Number of encoder layers (Fixed to 24).
        mlp_size (int): Size of the feedforward neural network in each layer (Fixed to 4096).
        
    """
    def __init__(self, image_size, patch_size, num_class):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_class=num_class,
            num_head=16,
            latent_size=1024,
            num_transformer=24,
            mlp_size=4096
        )


class ViTHuge(VisionTransformer):
    """
    ViTHuge is a variant of the Vision Transformer (ViT) architecture with customizable input size and number of classes.

    Args:
        image_size (int): The size of the input images.
        patch_size (int): The size of image patches for self-attention.
        num_class (int): The number of output classes for classification.

    Fixed Hyperparameters:
        num_head (int): Number of self-attention heads (Fixed to 16).
        latent_size (int): Dimensionality of latent space (Fixed to 1280).
        num_transformer (int): Number of encoder layers (Fixed to 32).
        mlp_size (int): Size of the feedforward neural network in each layer (Fixed to 5120).
        
    """
    def __init__(self, image_size, patch_size, num_class):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_class=num_class,
            num_head=16,
            latent_size=1280,
            num_transformer=32,
            mlp_size=5120
        )