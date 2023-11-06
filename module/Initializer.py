import sys
import tensorflow as tf

class Initializer:

    @staticmethod
    def mixed_precision16():
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    @staticmethod
    def memory_growth():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)

    @staticmethod
    def show_version():
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Python version: {sys.version}")
        