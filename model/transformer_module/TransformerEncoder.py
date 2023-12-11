from tensorflow.keras.layers import (
    Add,
    Dense,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)

class TransformerEncoder(Layer):
    """
    Custom layer implementing a Transformer Encoder block.

    Inherits from tf.keras.layers.Layer.

    Parameters:
        num_head (int): Number of attention heads.
        latent_size (int): Size of the latent space.
        mlp_size (int): Size of the feedforward layer.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Methods:
        build(input_shape): Build the layer by initializing the necessary components.
        call(input): Apply the layer to the input tensor, performing self-attention and feedforward operations.
        get_config(): Get the configuration of the layer.
        from_config(cls, config): Create an instance of the layer from a configuration dictionary.

    Example:
        ```python
        # Example usage in functional API
        # ... (previous layers)
        x = TransformerEncoder(num_head=8, latent_size=256, mlp_size=512)(previous_layer)
        # ... (add other layers)
        ```
    """
    num_instances = 0

    def __init__(self, num_head, latent_size, mlp_size, **kwargs):
        TransformerEncoder.num_instances += 1
        layer_name = f"Transformer_Encoder_{TransformerEncoder.num_instances}"
        super(TransformerEncoder, self).__init__(name=layer_name)
        self.num_head = num_head
        self.latent_size = latent_size
        self.mlp_size = mlp_size

    def build(self, input_shape):
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.multi_head = MultiHeadAttention(self.num_head, self.latent_size // self.num_head)
        self.mlp1 = Dense(self.mlp_size, activation="gelu")
        self.mlp2 = Dense(self.latent_size)
        super().build(input_shape)

    def call(self, input):
        x1 = self.layer_norm1(input)
        x1 = self.multi_head(x1, x1)
        x1 = Add()([x1, input])

        x2 = self.layer_norm2(x1)
        x2 = self.mlp1(x2)
        x2 = self.mlp2(x2)
        output = Add()([x1, x2])
        return output
    
    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'num_head': self.num_head,
            'latent_size': self.latent_size,
            'mlp_size': self.mlp_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)