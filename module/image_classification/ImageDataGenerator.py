import tensorflow as tf
from .DataProcessor import DataProcessor as processor
from .ImageAugmentation import ImageAugmentation

class ErrorHandler:
    """
    Error handling utility for input validation.

    Methods:
        validate_input(input): Validates the format of input data.
        validate_image_size(image_size): Validates the format of image_size.
        validate_translation(translate_range): Validates the format of translate_range.
        validate_rotation(rotation_range): Validates the format of rotation_range.

    Note: This class is designed for validating input parameters related to ImageDataGenerator class.
    """
    @staticmethod
    def validate_input(input):
        if not isinstance(input, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in input):
            raise ValueError("Invalid input. It should be a list of tuple (image file path, label string)")

    @staticmethod
    def validate_image_size(image_size):
        is_valid_tuple = isinstance(image_size, tuple) and len(image_size) == 2 and all(isinstance(dim, int) for dim in image_size)
        is_valid_int = isinstance(image_size, int)
        if not (is_valid_int or is_valid_tuple):
            raise ValueError("Invalid image_size. It should be an integer tuple (width, height) or an integer number")
    
    @staticmethod
    def validate_translation(translate_range):
        is_valid_tuple = isinstance(translate_range, tuple) and len(translate_range) == 2
        is_valid_float = isinstance(translate_range, float)
        if not (is_valid_tuple or is_valid_float or translate_range is None):
            raise ValueError("Invalid translate_range. It should be a tuple (float, float) or a float.")

    @staticmethod
    def validate_rotation(rotation_range):
        is_valid_int = isinstance(rotation_range, int) and rotation_range >= 0
        if not (is_valid_int or rotation_range is None):
            raise ValueError("Invalid rotation_range. It should be a positive integer")
        
class ImageDataGenerator:
    """
    Data generator for image classification tasks with preprocessing options such as image augmentation and validation split.

    Methods:
        generate_dataset_from_generator(batch_size, balance=False, train_drop_remainder=False): Generates a TensorFlow dataset using a generator.
        generate_dataset_from_tensor_slices(batch_size, balance=False, train_drop_remainder=False): Generates a TensorFlow dataset using `tf.data.Dataset.from_tensor_slices`.

    Example:
        ```python
        # Example usage of image data generator
        generator = ImageDataGenerator(
            input=data_list,
            image_size=(224, 224),
            keep_aspect_ratio=True,
            label_mode="onehot",
            horizontal_flip=True,
            vertical_flip=False,
            translate_range=(0.1, 0.1),
            rotation_range=20,
            border_method="constant",
            validation_split=0.2
        )

        train_dataset, test_dataset = generator.generate_dataset_from_generator(batch_size=32)
        ```

    Note: This class is designed to generate TensorFlow datasets for training and testing
    image classification models with various preprocessing options.
    """
    def __init__(
        self, 
        input,
        image_size, 
        keep_aspect_ratio=True,
        label_mode="onehot", # "onehot" or "index"
        horizontal_flip=False, 
        vertical_flip=False, 
        translate_range=None, 
        rotation_range=None,
        border_method="constant", # "constant" or "replicate"
        validation_split=0.0,
    ):
        # Error handler
        ErrorHandler.validate_input(input)
        ErrorHandler.validate_image_size(image_size)
        ErrorHandler.validate_translation(translate_range)
        ErrorHandler.validate_rotation(rotation_range)

        # Private artibute    
        self.__input = input
        self.__image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__horizontal_flip = horizontal_flip
        self.__label_mode=label_mode
        self.__vertical_flip = vertical_flip
        self.__translate_range = translate_range
        self.__rotation_range = rotation_range
        self.__border_method = border_method
        self.__validation_split = validation_split
        self.__summary(input, "Total")
        
    # Private methods
    def __summary(self, data_list, text):
        label_list = [label for _, label in data_list]
        label_set = sorted(set(label_list))
        print(f"{text}: {len(data_list)} dataset including")
        for label in label_set:
            counter = label_list.count(label)
            print(f"{label}: {counter}")
        
    def __image_reader_tf(self, image_path): 
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(
            images = image, 
            size = self.__image_size, 
            method = "bilinear", 
            preserve_aspect_ratio = self.__keep_aspect_ratio, 
            antialias = True,
        )
        image = image / 255.0
        return image
    
    @tf.autograph.experimental.do_not_convert
    def __train_preprocessing(self, image_path, label):
        image = self.__image_reader_tf(image_path)
        augmentation_list = []
        augment = ImageAugmentation(
            image_size = self.__image_size, 
            translate_range = self.__translate_range, 
            rotation_range = self.__rotation_range,
            border_method = self.__border_method,
        )

        if self.__translate_range is not None:
            augmentation_list.append(augment.translation)

        if self.__rotation_range is not None:
            augmentation_list.append(augment.rotation)

        if self.__horizontal_flip:
            augmentation_list.append(augment.horizontal_flip)

        if self.__vertical_flip:
            augmentation_list.append(augment.vertical_flip)

        augmentation_list.append(augment.padding)

        for func in augmentation_list:
            image = tf.numpy_function(func=func, inp=[image], Tout=tf.float32, name="Image_" + func.__name__)
        image = tf.ensure_shape(image, (self.__image_size[1], self.__image_size[0], 3))
        return image, label
    
    @tf.autograph.experimental.do_not_convert
    def __test_preprocessing(self, image_path, label):
        image = self.__image_reader_tf(image_path)

        augment = ImageAugmentation(
            image_size = self.__image_size, 
            translate_range = self.__translate_range, 
            rotation_range = self.__rotation_range,
            border_method = self.__border_method,
        )

        image = tf.numpy_function(func=augment.padding, inp=[image], Tout=tf.float32, name="Image_" + augment.padding.__name__)
        image = tf.ensure_shape(image, (self.__image_size[1], self.__image_size[0], 3))
        return image, label
    
    # Public methods
    def generate_dataset_from_generator(self, batch_size, balance=False, train_drop_remainder=False):
        def train_generator(data):
            for image_path, label in data:
                image, label = self.__train_preprocessing(image_path, label)
                
                yield image, label
        
        def test_generator(data):
            for image_path, label in data:
                image, label = self.__test_preprocessing(image_path, label)
                yield image, label
        
        if not isinstance(batch_size, int):
            raise ValueError("Invalid batch_size. It should be an integer.")
        
        data_list = processor.balance_data(self.__input) if balance else self.__input
        label_list = [label for _, label in data_list]
        label_set = list(sorted(set(label_list)))
        label_dict = processor.create_label_dict(label_set, self.__label_mode)
        train, test = processor.validation_spliter(data_list, label_set, self.__validation_split)
        
        # Print summary
        print("=====")
        class_weight = processor.calculate_class_weight(train)
        self.__summary(train, "Train")
        print(f"Class weight: {class_weight}")
        print("=====")
        self.__summary(test, "Test")  

        # Convert label string to one-hot-vector or label index
        train = [(image, label_dict[label]) for image, label in train]
        test = [(image, label_dict[label]) for image, label in test]
        
        # Create Train dataset
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_generator(train),
            output_signature=(tf.TensorSpec(shape=(self.__image_size[1], self.__image_size[0], 3,), dtype=tf.float32),
                              tf.TensorSpec(shape=(len(label_set),), dtype=tf.float32))
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=len(train))
        train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=train_drop_remainder)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Create Test dataset
        test_dataset = tf.data.Dataset.from_generator(
            lambda: test_generator(test),
            output_signature=(tf.TensorSpec(shape=(self.__image_size[1], self.__image_size[0], 3,), dtype=tf.float32),
                              tf.TensorSpec(shape=(len(label_set),), dtype=tf.float32))
        )
        test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, test_dataset

    def generate_dataset_from_tensor_slices(self, batch_size, balance=False, train_drop_remainder=False):
        if not isinstance(batch_size, int):
            raise ValueError("Invalid batch_size. It should be an integer.")
        
        data_list = processor.balance_data(self.__input) if balance else self.__input
        label_list = [label for _, label in data_list]
        label_set = list(sorted(set(label_list)))
        label_dict = processor.create_label_dict(label_set, self.__label_mode)

        train, test = processor.validation_spliter(data_list, label_set, self.__validation_split)
        
        # Print summary
        print("=====")
        class_weight = processor.calculate_class_weight(train)
        self.__summary(train, "Train")
        print(f"Class weight: {class_weight}")
        print("=====")
        self.__summary(test, "Test")      

        # Convert label string to one-hot-vector or label index
        train = [(image, label_dict[label]) for image, label in train]
        test = [(image, label_dict[label]) for image, label in test]

        # Create Train dataset
        if self.__validation_split < 1: # need condition to prevent error during mapping
            image_list = [image for image, _ in train]
            label_list = [label for _, label in train]
            train_dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))  
            train_dataset = train_dataset.map(self.__train_preprocessing)
            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
            train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=train_drop_remainder)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = None

        # Create Test dataset
        if self.__validation_split > 0: # need condition to prevent error during mapping
            image_list = [image for image, _ in test]
            label_list = [label for _, label in test]
            test_dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))  
            test_dataset = test_dataset.map(self.__test_preprocessing)
            test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            test_dataset = None

        return train_dataset, test_dataset