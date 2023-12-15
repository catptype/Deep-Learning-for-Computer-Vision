import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input

from .transformer.ImagePatcher import ImagePatcher
from .transformer.PatchEncoder import PatchEncoder
from .transformer.TransformerEncoder import TransformerEncoder
from .DeepLearningModel import DeepLearningModel

class CompactVisionTransformer(DeepLearningModel):
    """
    Custom model implementing a Compact Vision Transformer.

    Inherits from DeepLearningModel.

    Parameters:
        image_size (int): Size of the input images (assumed to be square).
        patch_size (int): Size of each image patch.
        num_class (int): Number of classes for classification.
        num_head (int): Number of attention heads in the Transformer encoder.
        latent_size (int): Dimensionality of the latent space in the Transformer encoder.
        num_transformer (int): Number of Transformer encoder blocks.
        mlp_size (int): Size of the feedforward layer in the Transformer encoder.

    Methods:
        build_model(): Build the Compact Vision Transformer model.

    Example:
        ```python
        # Example usage to create a CompactVisionTransformer model
        model = CompactVisionTransformer(
            image_size=224,
            patch_size=16,
            num_class=10,
            num_head=4,
            latent_size=256,
            num_transformer=6,
            mlp_size=512,
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