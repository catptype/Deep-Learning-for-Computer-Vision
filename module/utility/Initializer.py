import sys
import tensorflow as tf

class Initializer:
    """
    Utility class for initializing TensorFlow settings.

    Methods:
        mixed_precision16(): Sets the global policy for mixed precision to "mixed_float16".
        memory_growth(): Enables GPU memory growth for all available GPUs.
        set_gpu_memory_limit(MB=1024): Sets a specific memory limit (in megabytes) for GPU(s) and disables memory growth.
        show_version(): Displays the versions of TensorFlow and Python.

    Example:
        ```python
        # Example usage of Initializer class
        Initializer.mixed_precision16()
        Initializer.memory_growth()
        Initializer.set_gpu_memory_limit(MB=2048)
        Initializer.show_version()
        ```

    """
    @staticmethod
    def mixed_precision16():
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    @staticmethod
    def memory_growth():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)
    
    @staticmethod
    def set_gpu_memory_limit(MB=1024):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(MB))]
            )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    @staticmethod
    def show_version():
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Python version: {sys.version}")
        