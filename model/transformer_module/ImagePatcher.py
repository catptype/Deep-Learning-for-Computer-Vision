import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import Layer

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