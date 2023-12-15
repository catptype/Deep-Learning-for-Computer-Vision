import tensorflow as tf
from tensorflow.keras.layers import Layer

class ImagePatcher(Layer):
    """
    Custom layer to patch an input image into non-overlapping patches.

    Inherits from tf.keras.layers.Layer.

    Parameters:
        patch_size (int): Size of the square patches.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Methods:
        call(input): Apply the layer to the input tensor, extracting non-overlapping patches.
        get_config(): Get the configuration of the layer.
        from_config(cls, config): Create an instance of the layer from a configuration dictionary.

    Example:
        ```python
        # Example usage in functional API
        # ... (previous layers)
        patches = ImagePatcher(patch_size=16)(previous_layer)
        # ... (add other layers)
        ```
    """
    def __init__(self, patch_size, **kwargs):
        super(ImagePatcher, self).__init__(name="Image_Patcher")
        self.patch_size = patch_size
    
    def call(self, input):
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
    
    def get_config(self):
        config = super(ImagePatcher, self).get_config()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
