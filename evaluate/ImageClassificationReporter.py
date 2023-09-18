# Make python won’t try to write .pyc files on the import of source modules
import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Avoid out of memory errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Suppress log below than ERROR
# such INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
tf.get_logger().setLevel('ERROR')

class ImageClassificationReporter():
    """
    Utility class for generating classification reports from a trained image classification model.

    This class allows you to generate classification reports, including a confusion matrix,
    text-based classification report, and a visual representation of the confusion matrix.

    Args:
        h5_file (str): The path to the trained model in H5 format.
        image_list (list): List of image file paths for prediction.
        label_list (list): List of ground truth labels corresponding to the images.
        class_list (list): List of class labels.

    Attributes:
        confusion_matrix (numpy.ndarray): The confusion matrix.
        text_report (str): The text-based classification report.
        figure_report (matplotlib.figure.Figure): The visual representation of the confusion matrix.

    Methods:
        print_report(): Print the text-based classification report.
        export_report(): Export the visual representation of the confusion matrix as an image file.
    """
    def __init__(self, h5_file, image_list, label_list, class_list):
        """
        Initialize the ClassificationReportGenerator instance.

        Args:
            h5_file (str): The path to the trained model in H5 format.
            image_list (list): List of image file paths for prediction.
            label_list (list): List of ground truth labels corresponding to the images.
            class_list (list): List of class labels.
        """
        self.__h5_file = h5_file
        self.__image_list = image_list
        self.__label_list = label_list
        self.__class_list = class_list
        self.__predict_list = self.__execute()
        self.confusion_matrix = self.__generate_confusion_matrix()
        self.text_report = self.__generate_text_report()
        self.figure_report = self.__generate_figure_report()
    
    # Private methods
    def __execute(self):
        print(f"Loading model ... ", end="")
        model = load_model(self.__h5_file)
        print(f"Complete")        

        width = model.layers[0].input_shape[0][1]
        height = model.layers[0].input_shape[0][2]
        
        progress_bar = tf.keras.utils.Progbar(len(self.__image_list), 
                                              width=20, 
                                              interval=0.2, 
                                              unit_name='image')
        
        def preprocessing(image_path):
            image = tf.io.read_file(image_path)
            image = tf.io.decode_image(image, expand_animations = False)
            image = tf.image.resize(image, (width,height), preserve_aspect_ratio=True, antialias=True)
            image = tf.image.resize_with_pad(image,width,height)
            image = image / 255.0
            return image

        def predict(image_list):
            image = preprocessing(image_list)
            image = np.expand_dims(image.numpy(), axis=0)
            progress_bar.add(1)
            return model.predict(image)
        
        print("Processing class prediction ...")
        max_threads = os.cpu_count() // 2  # Half number of concurrent threads
        with ThreadPoolExecutor(max_threads) as executor:
            predictions_list = list(executor.map(predict, self.__image_list))
        
        predicted_class = np.concatenate(predictions_list)
        predicted_class = np.argmax(predicted_class, axis=1)
        predicted_class = [self.__class_list[idx] for idx in predicted_class]

        # Release memory resources explicitly
        tf.keras.backend.clear_session()

        return predicted_class

    def __generate_confusion_matrix(self):
        print("Generating confusion matrix ... ", end="")
        matrix = confusion_matrix(self.__label_list, self.__predict_list)
        print("Complete")
        return matrix
    
    def __generate_text_report(self):
        print("Generating text based classification report ... ", end="")
        report = classification_report(self.__label_list, self.__predict_list)
        print("Complete")
        return report
    
    def __generate_figure_report(self):
        print("Visualizing confusion matrix ... ", end="")
        if "/" in self.__h5_file:
            model_name = self.__h5_file.split('/')[-1].replace('.h5','')
        else:
            model_name = self.__h5_file.split('\\')[-1].replace('.h5','')

        # Creating a new figure
        fig = plt.figure(figsize=(8,6))
        
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\n{model_name}\n', fontsize=16)
        plt.colorbar()

        # Adding axis labels
        plt.xticks(np.arange(len(self.__class_list)), self.__class_list, rotation=0, fontsize=16)
        plt.yticks(np.arange(len(self.__class_list)), self.__class_list, fontsize=16)

        # Adding annotations (numbers) in the cells
        for i in range(len(self.__class_list)):
            for j in range(len(self.__class_list)):
                plt.text(j, i, str(self.confusion_matrix[i, j]), 
                        ha='center', 
                        va='center', 
                        backgroundcolor='white',
                        color='black',
                        fontsize=20,
                        fontweight=1000,
                        )

        plt.xlabel('\nPredicted', fontsize=16)
        plt.ylabel('Actual', fontsize=16)
        plt.tight_layout()
        print("Complete")
        return fig
    
    # Public methods
    def print_report(self):
        """
        Print the text-based classification report.
        """
        print("Classification Report")
        print(self.text_report)
    
    def export_report(self):
        """
        Export the visual representation of the confusion matrix as an image file.

        Note:
            - The image file will be saved with the same name as the model file (without the .h5 extension).
        """
        if "/" in self.__h5_file:
            filename = self.__h5_file.split('/')[-1].replace('.h5','')
        else:
            filename = self.__h5_file.split('\\')[-1].replace('.h5','')
        print(f"Export figure: {filename} ... ", end="")    
        self.figure_report.savefig(f'{filename}.jpg', bbox_inches="tight", facecolor="white")
        print("Complete")