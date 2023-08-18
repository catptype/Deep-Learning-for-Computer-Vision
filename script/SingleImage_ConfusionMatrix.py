# Make python won’t try to write .pyc files on the import of source modules
import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

# Avoid out of memory errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Suppress log below than ERROR
# such INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
tf.get_logger().setLevel('ERROR')

class SingleImage_ConfusionMatrix(): # Rename later TOO LONG!!
    def __init__(self, h5_file, image_list, label_list, class_list):
        self.h5_file = h5_file
        self.image_list = image_list
        self.label_list = label_list
        self.class_list = class_list
        self.predict_list = self.execute()
        self.confusion_matrix = self.matrix()
        self.report = self.text_report()
        self.figure = self.figure_report()
    
    def matrix(self):
        print("Generating confusion matrix ... ", end="")
        matrix = confusion_matrix(self.label_list, self.predict_list)
        print("Complete")
        return matrix
    
    def text_report(self):
        print("Generating text based classification report ... ", end="")
        report = classification_report(self.label_list, self.predict_list)
        print("Complete")
        return report
    
    def execute(self):
        
        print(f"Loading model ... ", end="")
        model = load_model(self.h5_file)
        print(f"Complete")        

        width = model.layers[0].input_shape[0][1]
        height = model.layers[0].input_shape[0][2]
        
        progress_bar = tf.keras.utils.Progbar(len(self.image_list), 
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
        max_threads = os.cpu_count()  # Maximum number of concurrent threads
        with ThreadPoolExecutor(max_threads) as executor:
            predictions_list = list(executor.map(predict, self.image_list))
        
        predicted_class = np.concatenate(predictions_list)
        predicted_class = np.argmax(predicted_class, axis=1)
        predicted_class = [self.class_list[idx] for idx in predicted_class]

        return predicted_class

    def print_report(self):
        print("Classification Report")
        print(self.report)
    
    def figure_report(self):
        print("Visualizing confusion matrix ... ", end="")
        if "/" in self.h5_file:
            model_name = self.h5_file.split('/')[-1].replace('.h5','')
        else:
            model_name = self.h5_file.split('\\')[-1].replace('.h5','')

        # Creating a new figure
        fig = plt.figure(figsize=(8,6))
        
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\n{model_name}\n', fontsize=16)
        plt.colorbar()

        # Adding axis labels
        plt.xticks(np.arange(len(self.class_list)), self.class_list, rotation=0, fontsize=16)
        plt.yticks(np.arange(len(self.class_list)), self.class_list, fontsize=16)

        # Adding annotations (numbers) in the cells
        for i in range(len(self.class_list)):
            for j in range(len(self.class_list)):
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
    
    def export_report(self):
        if "/" in self.h5_file:
            filename = self.h5_file.split('/')[-1].replace('.h5','')
        else:
            filename = self.h5_file.split('\\')[-1].replace('.h5','')
        print(f"Export figure: {filename} ... ", end="")    
        self.figure.savefig(f'{filename}.jpg')
        print("Complete")