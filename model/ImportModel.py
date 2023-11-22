import tensorflow as tf
from .DeepLearningModel import DeepLearningModel

class ImportModel(DeepLearningModel):
    """
    A class for importing and loading pre-trained deep learning models from .h5 files.

    Attributes:
        h5file (str): The path to the .h5 file containing the pre-trained model.

    Methods:
        __init__(self, h5file): Initialize the ImportModel instance with the path to the .h5 file.
        build_model(self): Load and build the pre-trained model, handling custom Keras layers if present.
    """
    def __init__(self, h5file):
        """
        Initialize an ImportModel instance.

        Parameters:
            h5file (str): The path to the .h5 file containing the pre-trained model.
        """
        if not h5file.endswith(".h5"):
            raise ValueError("Invalid: The h5file must have a .h5 extension.")
        
        self.h5file = h5file
        super().__init__()

    def build_model(self):
        """
        Load and build the pre-trained model, handling custom Keras layers if present.

        If the model contains custom Keras layers, this method loads the model with the necessary custom objects.

        Returns:
            tf.keras.Model: The loaded pre-trained model.
        """
        try:
            print("Loading model ...")
            model = tf.keras.models.load_model(self.h5file, compile=False)
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

            model = tf.keras.models.load_model(self.h5file, custom_objects=custom_objects, compile=False)
            print("Complete")
        
        return model