import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization

from .transformer.ClassToken import ClassToken
from .transformer.ImagePatcher import ImagePatcher
from .transformer.PatchEncoder import PatchEncoder
from .transformer.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel


class VisionTransformer(DeepLearningModel):
    """
    Custom model implementing the Vision Transformer (ViT) architecture.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        patch_size (int): Size of the image patches.
        num_class (int): Number of classes for classification.
        num_head (int): Number of attention heads in the transformer.
        latent_size (int): Size of the latent vectors in the transformer.
        num_transformer (int): Number of transformer blocks in the architecture.
        mlp_size (int): Size of the Multi-Layer Perceptron (MLP) in the transformer.

    Methods:
        build_model(): Build the Vision Transformer model.

    Subclasses:
        - ViTBase
        - ViTLarge
        - ViTHuge

    Example:
        ```python
        # Example usage to create a VisionTransformer model
        model = VisionTransformer(
            image_size=224, 
            patch_size=16, 
            num_class=10, 
            num_head=12, 
            latent_size=768,     
            num_transformer=12,        
            mlp_size=3072,
        )    
        ```
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
    Subclass of VisionTransformer with specific configuration.

    Inherits from VisionTransformer.

    Example:
        ```python
        # Example usage to create a ViTBase model
        model = ViTBase(image_size=224, patch_size=16, num_class=10)
        ```

    Note: Inherits parameters from VisionTransformer.
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
    Subclass of VisionTransformer with specific configuration.

    Inherits from VisionTransformer.

    Example:
        ```python
        # Example usage to create a ViTLarge model
        model = ViTLarge(image_size=224, patch_size=16, num_class=10)
        ```

    Note: Inherits parameters from VisionTransformer.
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
    Subclass of VisionTransformer with specific configuration.

    Inherits from VisionTransformer.

    Example:
        ```python
        # Example usage to create a ViTHuge model
        model = ViTHuge(image_size=224, patch_size=16, num_class=10)
        ```

    Note: Inherits parameters from VisionTransformer.
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