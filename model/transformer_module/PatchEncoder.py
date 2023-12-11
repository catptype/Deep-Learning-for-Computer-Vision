import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Layer

class PatchEncoder(Layer):
    """
    Custom layer to encode patches with linear projection and positional embedding.

    Inherits from tf.keras.layers.Layer.

    Parameters:
        num_patch (int): Number of patches.
        latent_size (int): Size of the latent space.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Methods:
        build(input_shape): Build the layer by initializing the linear projection and positional embedding.
        call(input): Apply the layer to the input tensor, performing linear projection and positional embedding.
        get_config(): Get the configuration of the layer.
        from_config(cls, config): Create an instance of the layer from a configuration dictionary.

    Example:
        ```python
        # Example usage in functional API
        # ... (previous layers)
        x = PatchEncoder(num_patch=64, latent_size=256)(previous_layer)
        # ... (add other layers)
        ```
    """
    def __init__(self, num_patch, latent_size, **kwargs):
        super(PatchEncoder, self).__init__(name="Patch_Encoder")
        self.num_patch = num_patch
        self.latent_size = latent_size

    def build(self, input_shape):
        self.linear_projection = Dense(self.latent_size)
        self.positional_embedding = Embedding(self.num_patch, self.latent_size)
        super().build(input_shape)

    def call(self, input):
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