import sys
sys.dont_write_bytecode = True

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Dense,
    Embedding,
    Flatten,
    Input,
    Layer,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
)
import tensorflow as tf
from .DeepLearningModel import DeepLearningModel


class ClassToken(Layer):
    """
    The ClassToken layer.

    This layer adds a class token to the input tensor. The class token is a
    learnable parameter used in Vision Transformers to represent global information.

    """
    def __init__(self):
        super(ClassToken, self).__init__(name="ClassToken")
    
    def build(self, input_shape):
        """
        Builds the ClassToken layer by initializing the class token weight.

        Args:
            input_shape (tuple): The shape of the input tensor.

        """
        init_w = tf.random_normal_initializer()
        self.init_weight = tf.Variable(
            initial_value = init_w(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True,
        )
        super().build(input_shape)
    
    def call(self, input):
        """
        Adds the class token to the input tensor.

        Args:
            input (tf.Tensor): The input tensor to which the class token is added.

        Returns:
            tf.Tensor: The input tensor with the class token added as the first element along the axis 1.
        """
        c_token = tf.broadcast_to(self.init_weight, [tf.shape(input)[0], 1, self.init_weight.shape[-1]])
        c_token = tf.cast(c_token, dtype=input.dtype)

        output = Concatenate(axis=1)([c_token, input])
        return output


class ConvToken(Layer):
    """
    ConvToken - A custom Keras layer for creating convolutional token embeddings.

    This class defines a custom Keras layer that takes a list of convolutional layer configurations
    and applies them sequentially to the input tensor. Each convolutional layer is followed by
    max-pooling, and the final output is a reshaped tensor representing token embeddings.

    Attributes:
        conv_layer (list): A list of integers representing the number of features for each
                        convolutional layer.

    """
    def __init__(self, conv_layer):
        """
        Constructor method for initializing the ConvToken layer.

        Args:
            conv_layer (list): A list of integers representing the number of features
                               for each convolutional layer.
        """
        self.conv_layer = conv_layer
        super(ConvToken, self).__init__(name="ConvToken")
    
    def build(self, input_shape):
        """
        Method for building the ConvToken layer with the specified input shape.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Note:
            This method creates a list of Conv2D layers based on the configuration provided
            in the `conv_layer` attribute.

        """
        self.conv2d_list = [Conv2D(num_feature, (3,3), padding="same", activation='relu', kernel_initializer="he_normal") for num_feature in self.conv_layer]
        super().build(input_shape)
    
    def call(self, input):
        """
        Method for performing the forward pass of the ConvToken layer.

        Args:
            input (tensor): The input tensor.

        Returns:
            output (tensor): The output tensor after applying convolutional layers and max-pooling.
        """
        for idx, _ in enumerate(self.conv_layer):
            if idx == 0:
                x = self.conv2d_list[idx](input)
            else:
                x = self.conv2d_list[idx](x)
            x = MaxPooling2D()(x)
        num_patch = x.shape[1] * x.shape[2]
        output = tf.reshape(x, (-1, num_patch, x.shape[-1]))
        return output


class ImagePatcher(Layer):
    """
    The ImagePatcher layer.

    This layer extracts patches from an input image tensor. These patches are
    used as the input to a Vision Transformer model.

    Args:
        patch_size (int): The size of each square patch to be extracted from the input image.

    """
    def __init__(self, patch_size):
        self.patch_size = patch_size
        super(ImagePatcher, self).__init__(name="Image_Patcher")
    
    def call(self, input):
        """
        Extracts patches from an input image tensor.

        Args:
            input (tf.Tensor): The input image tensor from which patches will be extracted.

        Returns:
            tf.Tensor: A tensor containing extracted patches from the input image. The shape
                of the returned tensor will be (batch_size, num_patches, patch_size * patch_size * channels).
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
    The PatchEncoder layer.

    This layer encodes patches extracted from an image into a latent space
    using a linear projection and positional embeddings.

    Args:
        num_patch (int): The number of patches to be encoded.
        latent_size (int): The size of the latent space for encoding.

    """
    def __init__(self, num_patch, latent_size):
        self.num_patch = num_patch
        self.latent_size = latent_size
        super(PatchEncoder, self).__init__(name="Patch_Encoder")

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


class ViTModel(DeepLearningModel):
    """
    Vision Transformer (ViT) model.

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
    

class CVTModel(DeepLearningModel):
    """
    CVTModel - A custom deep learning model for Compact Vision Transformers (CVT).

    This class defines a custom deep learning model for Compact Vision Transformers (CVT)
    that consists of an image patcher, patch encoder, transformer encoder, sequence pooling, and a classification head.

    Attributes:
        image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
        patch_size (int): The size of image patches.
        num_classes (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the transformer.
        latent_size (int): The latent dimension size for patch embeddings.
        num_layer (int): The number of transformer encoder layers.
        mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.

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
        """
        Constructor method for initializing the CVTModel.

        Args:
            image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
            patch_size (int): The size of image patches.
            num_classes (int): The number of output classes for classification.
            num_head (int): The number of attention heads in the transformer.
            latent_size (int): The latent dimension size for patch embeddings.
            num_layer (int): The number of transformer encoder layers.
            mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
        """
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
        for _ in range(self.num_layer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # SeqPool
        SeqPool = Dense(1, activation="softmax")(x)        
        SeqPool = tf.matmul(SeqPool, x, transpose_a=True)

        # Classification
        x = Flatten()(SeqPool)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model_name = f"CVT_I{self.image_size}x{self.image_size}_P{self.patch_size}_L{self.num_layer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_classes}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model


class CCTModel(DeepLearningModel):
    """
    CCTModel - A custom deep learning model for Compact Convolutional Transformer (CCT).

    This class defines a custom deep learning model for Compact Convolutional Transformer (CCT)
    that combines convolutional tokenization, position embeddings, transformer encoding, sequence pooling,
    and a classification head.

    Attributes:
        image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
        conv_layers (list): A list of integers representing the number of features for each convolutional layer.
        num_classes (int): The number of output classes for classification.
        num_head (int): The number of attention heads in the transformer.
        latent_size (int): The latent dimension size for patch embeddings.
        num_layer (int): The number of transformer encoder layers.
        mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
        position_embedding (bool): A flag indicating whether to include position embeddings.

    """
    def __init__(
        self,
        image_size,
        conv_layer,
        num_classes,
        num_head,
        latent_size,
        num_layer,
        mlp_size,
        position_embedding=False,
    ):
        """
        Constructor method for initializing the CCTModel.

        Args:
            image_size (int): The size of the input image (e.g., 224 for a 224x224 image).
            conv_layer (list): A list of integers representing the number of features
                               for each convolutional layer.
            num_classes (int): The number of output classes for classification.
            num_head (int): The number of attention heads in the transformer.
            latent_size (int): The latent dimension size for patch embeddings.
            num_layer (int): The number of transformer encoder layers.
            mlp_size (int): The size of the multi-layer perceptron (MLP) in the transformer.
            position_embedding (bool): A flag indicating whether to include position embeddings.
        """
        assert latent_size % num_head == 0, f"Latent size ({latent_size}) is not divisible by number of attention heads ({num_head})"

        self.conv_layer = conv_layer
        self.position_embedding = position_embedding
        self.num_head = num_head
        self.latent_size = latent_size
        self.num_layer = num_layer
        self.mlp_size = mlp_size
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        """
        Method for building the CCTModel architecture and returning the Keras model.

        Returns:
            model (Model): The Keras model representing the CCTModel.
        """
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # Convolution Token
        x = ConvToken(conv_layer=self.conv_layer)(input)
        assert self.latent_size == x.shape[-1], f"Convolutional Tokenizer must has channel ({x.shape[-1]}) = embedding dimension ({self.latent_size})"
        
        # Patch encoder
        if self.position_embedding:
            x = PatchEncoder(num_patch=x.shape[1], latent_size=self.latent_size)(x)

        # Transformer encoder
        for _ in range(self.num_layer):
            x = TransformerEncoder(num_head=self.num_head, latent_size=self.latent_size, mlp_size=self.mlp_size)(x)

        # SeqPool
        SeqPool = Dense(1, activation="softmax")(x)        
        SeqPool = tf.matmul(SeqPool, x, transpose_a=True)

        # Classification
        x = Flatten()(SeqPool)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model_name = f"CCT_I{self.image_size}x{self.image_size}_Conv{self.conv_layer}_L{self.num_layer}_H{self.num_head}_D{self.latent_size}_MLP{self.mlp_size}_{self.num_classes}Class"
        model = Model(inputs=[input], outputs=output, name=model_name)
        return model
