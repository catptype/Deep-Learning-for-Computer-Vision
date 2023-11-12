import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model.ImportModel import ImportModel
from sklearn.metrics import confusion_matrix, classification_report

# Suppress log below than ERROR
# such INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
tf.get_logger().setLevel('ERROR')

class ConfusionMatrix:
    """
    Image Classification Reporter.

    This class provides a reporting mechanism for evaluating image classification models.
    It takes a pre-trained model, input data, and labels, and generates classification reports,
    including a confusion matrix and a visual representation of the confusion matrix.

    Attributes:
        model_name (str): The name of the loaded model.
        confusion_matrix (numpy.ndarray): The confusion matrix generated from predictions.
        text_report (str): The text-based classification report.
        figure_report (matplotlib.figure.Figure): The visual representation of the confusion matrix as a matplotlib figure.

    """
    def __init__(self, h5_file, data_list, label_list, model_name=None, force_batch=None):
        """
        Initialize the ImageClassificationReporter instance.

        Args:
            h5_file (str): The file path of the pre-trained image classification model in ".h5" format.
            data_list (list): A list of tuples containing image file paths and their corresponding labels.
            label_list (list): A list of labels for classification.
            model_name (str, optional): An optional name for the model. If not provided, the name from the loaded model is used.
            force_batch (int, optional): An optional batch size for predictions. If provided, predictions are made in batches.
        """
        if not h5_file.endswith(".h5"):
            raise ValueError("Invalid: The h5_file must have a .h5 extension.")
        
        self.__model = ImportModel(h5_file)
        self.model_name = self.__model.model.name if model_name is None else model_name
        self.__input_shape = self.__model.get_input_shape()
        self.__image_path = [image for image, _ in data_list]
        self.__true_label = [label for _, label in data_list]
        self.__label_list = label_list
        self.__predict_label = self.__predict(32) if force_batch is None else self.__predict(force_batch)
        self.confusion_matrix = self.__generate_confusion_matrix()
        self.text_report = self.__generate_text_report()
        self.figure_report = self.__visualize_report()

    # Private methods
    @tf.autograph.experimental.do_not_convert
    def __image_resize(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(
            image, 
            (self.__input_shape[0], self.__input_shape[1]),
            method = "bilinear",
            preserve_aspect_ratio = True,
            antialias = True,
        )
        image = tf.image.resize_with_pad(image, self.__input_shape[0], self.__input_shape[1])
        image = image / 255.0
        
        return image

    def __predict(self, batch_size):
        print("Processing class prediction ...")
        prediction_result = []

        batch_image = tf.data.Dataset.from_tensor_slices(self.__image_path)
        batch_image = batch_image.map(self.__image_resize)
        batch_image = batch_image.batch(batch_size)

        progress_bar = tf.keras.utils.Progbar(
            len(batch_image), 
            width=20, 
            interval=0.2, 
            unit_name='batch',
        )
        
        for batch in batch_image:
            prediction = self.__model.predict(batch)
            prediction = np.argmax(prediction, axis=-1)
            prediction_result.extend(prediction)
            progress_bar.add(1)

        prediction_result = [self.__label_list[idx] for idx in prediction_result]

        # Release memory after finish prediction
        tf.keras.backend.clear_session()

        return prediction_result

    def __generate_confusion_matrix(self):
        print("Generating confusion matrix ... ", end="")
        matrix = confusion_matrix(self.__true_label, self.__predict_label)
        print("Complete")
        return matrix

    def __generate_text_report(self):
        print("Generating text based classification report ... ", end="")
        report = classification_report(self.__true_label, self.__predict_label)
        print("Complete")
        return report

    def __visualize_report(self):
        # Creating a new figure
        fig = plt.figure(figsize=(8,6))
        
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\n{self.model_name}\n', fontsize=16)
        plt.colorbar()

        # Adding axis labels
        plt.xticks(np.arange(len(self.__label_list)), self.__label_list, rotation=0, fontsize=16)
        plt.yticks(np.arange(len(self.__label_list)), self.__label_list, fontsize=16)

        # Adding annotations (numbers) in the cells
        for i in range(len(self.__label_list)):
            for j in range(len(self.__label_list)):
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
        """
        print(f"Export figure: {self.model_name} ... ", end="")    
        self.figure_report.savefig(f'{self.model_name}.jpg', bbox_inches="tight", facecolor="white")
        print("Complete")