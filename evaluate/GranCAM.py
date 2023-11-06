import sys
sys.dont_write_bytecode = True

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class GranCAM:
    """
    Class for generating Gradient-weighted Class Activation Maps (Grad-CAM) for a given model and image.
    Note: Incompatible with models using mixed precision float16.
    """
    def __init__(self, model):
        """
        Initializes the GranCAM instance.

        Parameters:
            model (str or tf.keras.Model): The pre-trained model or the path to a saved model (.h5).
        """
        self.model = self.__build_model(model)
    
    # Private methods
    def __build_model(self, model):
        if isinstance(model, str) and model.endswith(".h5"):
            print("Load model ... ", end="")
            model = load_model(model)
            print("Complete")

        conv2d_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        layer_name = conv2d_layers[-1].name

        print("Generate GranCAM model ... ", end="")
        conv2d_output = model.get_layer(layer_name).output
        model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv2d_output, model.output])
        print("Complete")
        return model
    
    def __get_epsilon(self):
        epsilon = 1
        while 1 + epsilon != 1:
            epsilon /= 2.        
        return epsilon 
    
    # Public methods
    def generate_heatmap(self, image, class_index=None, overlay=False):
        """
        Generates the Gradient-weighted Class Activation Map (Grad-CAM) for a given image.

        Parameters:
            image (np.ndarray): The input image as a NumPy array.
            class_index (int): The index of the class for which the heatmap is generated.
                              If None, the class with the highest predicted probability is used.
            overlay (bool): If True, overlays the heatmap on the original image.

        Returns:
            np.ndarray: The generated heatmap or the overlayed image, depending on the 'overlay' parameter.
        """
        # Get original image size for resizing heatmap
        width = image.shape[0]
        height = image.shape[1]

        # Expand image dimension from 3D to 4D
        image_4D = np.expand_dims(image.numpy(), axis=0)
        
        # Calculate gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.model(image_4D)
            if class_index is None:
                class_index = np.argmax(predictions)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        epsilon = self.__get_epsilon()
        conv_outputs = conv_outputs[0]
        
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0 + epsilon)
        heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (width, height))
        heatmap = plt.cm.coolwarm(heatmap)[:, :, :3]

        # Overlay with image if overlay boolean is True
        if overlay:
            alpha = 0.5
            image_array = np.array(image)
            overlay_heatmap = (1 - alpha) * image_array + alpha * heatmap
            return overlay_heatmap

        return heatmap