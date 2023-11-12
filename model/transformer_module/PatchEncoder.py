import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Layer,
)

class PatchEncoder(Layer):
    """
    The PatchEncoder layer.

    This layer encodes patches extracted from an image into a latent space
    using a linear projection and positional embeddings.

    Args:
        num_patch (int): The number of patches to be encoded.
        latent_size (int): The size of the latent space for encoding.

    """
    def __init__(self, num_patch, latent_size, **kwargs):
        super(PatchEncoder, self).__init__(name="Patch_Encoder")
        self.num_patch = num_patch
        self.latent_size = latent_size

    def build(self, input_shape):
        """
        Builds the PatchEncoder layer by initializing its sublayers.

        Args:
            input_shape (tuple): The shape of the input tensor.

        """
        self.linear_projection = Dense(self.latent_size)
        self.positional_embedding = Embedding(self.num_patch, self.latent_size)
        super().build(input_shape)

    def call(self, input):
        """
        Encodes patches into a latent space using linear projection and positional embedding.

        Args:
            input (tf.Tensor): The input tensor containing extracted patches.

        Returns:
            tf.Tensor: A tensor containing the encoded patches in the latent space.
        """
        # Linear projection and Positional embedding
        embedding_input = tf.range(start=0, limit=self.num_patch, delta=1)
        output = self.linear_projection(input) + self.positional_embedding(embedding_input)

        return output
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            'num_patch': self.num_patch,
            'latent_size': self.latent_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)