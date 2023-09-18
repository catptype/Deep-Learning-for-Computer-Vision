import sys
sys.dont_write_bytecode = True

from tensorflow.keras.layers import (
    Add,
    Dense,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)

class TransformerEncoder(Layer):
    """
    TransformerEncoder layer.

    This layer represents one encoder block of a Vision Transformer, which consists of
    multi-head self-attention and feedforward neural networks.

    Args:
        num_head (int): The number of attention heads in the multi-head self-attention mechanism.
        latent_size (int): The size of the latent space for the encoder.
        mlp_size (int): The size of the feedforward neural network hidden layer.

    """
    num_instances = 0

    def __init__(self, num_head, latent_size, mlp_size):
        self.num_head = num_head
        self.latent_size = latent_size
        self.mlp_size = mlp_size
        TransformerEncoder.num_instances += 1
        layer_name = f"Transformer_Encoder_{TransformerEncoder.num_instances}"
        super(TransformerEncoder, self).__init__(name=layer_name)

    def build(self, input_shape):
        """
        Builds the TransformerEncoder layer by initializing its sublayers.

        Args:
            input_shape (tuple): The shape of the input tensor.

        """
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.multi_head = MultiHeadAttention(self.num_head, self.latent_size // self.num_head)
        self.mlp1 = Dense(self.mlp_size, activation="gelu")
        self.mlp2 = Dense(self.latent_size)
        super().build(input_shape)

    def call(self, input):
        """
        Applies the TransformerEncoder layer to the input tensor.

        Args:
            input (tf.Tensor): The input tensor to be processed by the encoder.

        Returns:
            tf.Tensor: The output tensor after passing through the encoder block.
        """
        x1 = self.layer_norm1(input)
        x1 = self.multi_head(x1, x1)
        x1 = Add()([x1, input])

        x2 = self.layer_norm2(x1)
        x2 = self.mlp1(x2)
        x2 = self.mlp2(x2)
        output = Add()([x1, x2])
        return output