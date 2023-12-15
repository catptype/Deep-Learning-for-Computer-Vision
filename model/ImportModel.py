import tensorflow as tf
from .DeepLearningModel import DeepLearningModel

class ImportModel(DeepLearningModel):
    """
    A class for importing pre-trained deep learning models from a saved HDF5 file (.h5).

    This class extends the DeepLearningModel base class and provides a method to build the model
    from the specified HDF5 file. If the model contains custom Keras layers, it attempts to load
    them using the appropriate custom objects.

    Parameters:
    - h5file (str): The path to the HDF5 file containing the pre-trained model.

    Attributes:
    - h5file (str): The path to the HDF5 file containing the pre-trained model.

    Methods:
    - build_model(): Attempts to load the pre-trained model from the HDF5 file and handles
      the case of custom Keras layers, loading them with specified custom objects if necessary.

    Example:
    ```python
    # Example usage of ImportModel class
    model = ImportModel("path/to/pretrained_model.h5")
    ```

    Note:
    The custom objects for handling Keras layers are imported from specific modules within
    the package using relative imports. Ensure that the necessary modules are accessible.
    """
    def __init__(self, h5file):
        if not h5file.endswith(".h5"):
            raise ValueError("Invalid: The h5file must have a .h5 extension.")
        
        self.h5file = h5file
        super().__init__()

    def build_model(self):
        try:
            print("Loading model ...")
            model = tf.keras.models.load_model(self.h5file, compile=False)
            print("Model loaded successfully. No custom Keras layers detected.")
        except ValueError:
            print("ValueError: Custom Keras layers detected.\nLoading model with custom objects ...", end="")
            from .transformer.ClassToken import ClassToken
            from .transformer.ConvToken import ConvToken
            from .transformer.ImagePatcher import ImagePatcher
            from .transformer.PatchEncoder import PatchEncoder
            from .transformer.TransformerEncoder import TransformerEncoder
            
            custom_objects = {
                'ImagePatcher': ImagePatcher,
                'PatchEncoder': PatchEncoder,
                'ClassToken': ClassToken,
                'ConvToken': ConvToken,
                'TransformerEncoder': TransformerEncoder,
            }

            model = tf.keras.models.load_model(self.h5file, custom_objects=custom_objects, compile=False)
            print("COMPLETE")
        
        return model