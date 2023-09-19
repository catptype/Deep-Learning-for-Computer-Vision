import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from .DeepLearningModel import DeepLearningModel

class ImportModel(DeepLearningModel):
    """
    Deep learning model class for importing pre-trained models from H5 files.

    Args:
        h5file (str): The path to the H5 file containing the pre-trained model.

    Attributes:
        h5file (str): The path to the H5 file containing the pre-trained model.

    Methods:
        build_model(): Load the pre-trained model from the H5 file and set image_size and num_classes.
    """
    def __init__(self, h5file):
        """
        Initialize the ImportModel instance.

        Args:
            h5file (str): The path to the H5 file containing the pre-trained model.

        Raises:
            ValueError: If h5file does not have a valid .h5 file extension.
        """
        if not h5file.endswith(".h5"):
            raise ValueError("Invalid: The h5file must have a .h5 extension.")
        
        self.h5file = h5file
        super().__init__(image_size=None, num_classes=None)

    def build_model(self):
        """
        Load the pre-trained model from the H5 file and set image_size and num_classes.

        Returns:
            tf.keras.Model: The pre-trained model loaded from the H5 file.
        """
        try:
            print("Loading model ...")
            model = tf.keras.models.load_model(self.h5file)
            print("Model loaded successfully. No custom Keras layers detected.")
        except ValueError:
            print("Error: Custom Keras layers detected.\nLoading model with custom objects ...", end="")
            from .transformer_module.ClassToken import ClassToken
            from .transformer_module.ConvToken import ConvToken
            from .transformer_module.ImagePatcher import ImagePatcher
            from .transformer_module.PatchEncoder import PatchEncoder
            from .transformer_module.TransformerEncoder import TransformerEncoder
            
            custom_objects = {
                'ImagePatcher': ImagePatcher,
                'PatchEncoder': PatchEncoder,
                'ClassToken': ClassToken,
                'ConvToken': ConvToken,
                'TransformerEncoder': TransformerEncoder,
            }

            model = tf.keras.models.load_model(self.h5file, custom_objects=custom_objects)
            print("Complete")

        # Override image_size and num_classes
        self.image_size = model.layers[0].input_shape[0][1]
        self.num_classes = model.layers[-1].output_shape[-1]
        
        return model       