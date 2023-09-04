import sys
sys.dont_write_bytecode = True

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Add,
    Dense,
    Embedding,
    Flatten,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
import tensorflow as tf
from .DeepLearningModel import DeepLearningModel

class ImagePatcher(Layer):
    """
    ImagePatcher: Splits the input image into patches.

    Args:
        patch_size (int): Size of each image patch.
    
    """
    def __init__(self, patch_size):
        self.patch_size = patch_size
        super(ImagePatcher, self).__init__(name="Image_Patcher")
    
    def call(self, input):
        """
        Applies image patching to the input image.

        Args:
            input (tensor): Input image tensor.

        Returns:
            image_patch (tensor): Patches extracted from the input image.

        """                
        image_patch = tf.image.extract_patches(
            images=input,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        ) 
        num_patch = image_patch.shape[1] * image_patch.shape[2]
        image_patch = tf.reshape(image_patch, (-1, num_patch, image_patch.shape[-1]))
        return image_patch


class PatchEncoder(Layer):
    """
    PatchEncoder layer: Encodes image patches and applies linear projection with positional embedding.

    Args:
        num_patch (int): Number of image patches.
        latent_size (int): Size of the latent space.

    """
    def __init__(self, num_patch, latent_size):
        self.num_patch = num_patch
        self.latent_size = latent_size
        super(PatchEncoder, self).__init__(name="Patch_Encoder")

    def build(self, input_shape):
        """
        Builds the PatchEncoder layer by creating necessary sub-layers.

        Args:
            input_shape (tuple): Shape of the input tensor.

        """
        self.linear_projection = Dense(self.latent_size)
        self.positional_embedding = Embedding(self.num_patch, self.latent_size)
        super().build(input_shape)

    def call(self, input):
        """
        Applies patch encoding to the input patches.

        Args:
            input (tensor): Input patches tensor.

        Returns:
            output (tensor): Encoded patches with positional embeddings.

        """
        # Linear projection and Positional embedding
        embedding_input = tf.range(start=0, limit=self.num_patch, delta=1)
        output = self.linear_projection(input) + self.positional_embedding(embedding_input)
        return output


class TransformerEncoder(Layer):
    """
    TransformerEncoder layer: Applies multi-head self-attention and feed-forward layers.

    Args:
        num_head (int): Number of attention heads.
        latent_size (int): Size of the latent space.

    """
    num_instances = 0

    def __init__(self, num_head, latent_size):
        self.num_head = num_head
        self.latent_size = latent_size
        TransformerEncoder.num_instances += 1
        layer_name = f"Transformer_Encoder_{TransformerEncoder.num_instances}"
        super(TransformerEncoder, self).__init__(name=layer_name)

    def build(self, input_shape):
        """
        Builds the TransformerEncoder layer by creating necessary sub-layers.

        Args:
            input_shape (tuple): Shape of the input tensor.

        """
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.multi_head = MultiHeadAttention(self.num_head, self.latent_size)
        self.mlp1 = Dense(self.latent_size, activation="gelu")
        self.mlp2 = Dense(self.latent_size, activation="gelu")
        super().build(input_shape)

    def call(self, input):
        """
        Applies the TransformerEncoder layer to the input tensor.

        Args:
            input (tensor): Input tensor.

        Returns:
            output (tensor): Transformed tensor after multi-head self-attention and feed-forward layers.

        """
        x1 = self.layer_norm1(input)
        x1 = self.multi_head(x1, x1)
        x1 = Add()([x1, input])

        x2 = self.layer_norm2(x1)
        x2 = self.mlp1(x2)
        x2 = self.mlp2(x2)
        output = Add()([x1, x2])
        return output


class ViTModel(DeepLearningModel):
    """
    ViTModel: Vision Transformer model for image classification.

    Args:
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        num_classes (int): Number of classes in the classification task.
        num_head (int): Number of attention heads.
        latent_size (int): Size of the latent space.
        num_layer (int): Number of transformer layers.
        mlp_size (int): Size of the multi-layer perceptron.

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

        self.patch_size = patch_size
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_layer = num_layer
        self.mlp_size = mlp_size
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Builds the ViTModel architecture.

        Returns:
            model (tensorflow.keras.Model): Compiled ViT model.

        """       
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Image patcher
        x = ImagePatcher(patch_size=self.patch_size)(input)

        # Patch encoder
        x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Transformer encoder
        for _ in range(self.num_layer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size)(x)

        # Classification
        x = Flatten()(x)
        x = Dense(self.mlp_size, activation="gelu")(x)
        x = Dense(self.mlp_size, activation="gelu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model_name = f"ViT_L{self.num_layer}_I{self.image_size}x{self.image_size}_P{self.patch_size}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_classes}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model