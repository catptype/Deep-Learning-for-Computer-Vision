import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel

class YOLOv3(DeepLearningModel):
    """
    YOLOv3: You Only Look Once Version 3.

    This class defines the YOLOv3 deep learning model for object detection.

    Parameters:
        image_size (int or tuple): The input image size as an integer for square images
            or a tuple (width, height) for non-square images.
        num_anchor (int): The number of anchor boxes used for object detection.
        num_class (int): The number of output classes for object detection.
    """
    def __init__(self, image_size, num_anchor, num_class):
        """
        Initialize the YOLOv3 model with the specified parameters.

        Parameters:
            image_size (int or tuple): The input image size as an integer for square images
                or a tuple (width, height) for non-square images.
            num_anchor (int): The number of anchor boxes used for object detection.
            num_class (int): The number of output classes for object detection.
        """
        self.image_size = image_size
        self.num_class = num_class
        self.num_anchor = num_anchor
        super().__init__()

    def conv2D_block(self, input, num_feature, kernel=3, strides=1, upsampling=False):
        x = Conv2D(num_feature, (kernel, kernel), strides=strides, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        if upsampling:
            x = UpSampling2D()(x)
        return x

    def res_unit(self, input, num_feature):
        x = self.conv2D_block(input, num_feature // 2, kernel=1)
        x = self.conv2D_block(x, num_feature, kernel=3)
        x = Add()([input, x])
        return x
    
    def conv2D_unit(self, input, num_feature):
        x = self.conv2D_block(input, num_feature // 2, kernel=1)
        x = self.conv2D_block(x, num_feature)
        x = self.conv2D_block(x, num_feature // 2, kernel=1)
        x = self.conv2D_block(x, num_feature)
        x = self.conv2D_block(x, num_feature // 2, kernel=1)
        return x
    
    def scale_prediction(self, input, num_feature, name=None):
        output_feature = (self.num_class + 5) * 3
        x = self.conv2D_block(input, num_feature)
        x = Conv2D(output_feature, (1,1), activation="linear", dtype=tf.float32)(x)
        x = Reshape((x.shape[1], x.shape[2], self.num_anchor, 5 + self.num_class), name=name, dtype=tf.float32)(x)
        return x
        
    def build_model(self):
        # Input layer
        if isinstance(self.image_size, tuple) and len(self.image_size) == 2:
            input = Input(shape=(self.image_size[1], self.image_size[0], 3), name="Input_image")
            model_name = f"YOLOv3_{self.image_size[0]}x{self.image_size[1]}_a{self.num_anchor}_{self.num_class}Class"
        
        elif isinstance(self.image_size, int):
            input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")
            model_name = f"YOLOv3_{self.image_size}x{self.image_size}_a{self.num_anchor}_{self.num_class}Class"
        
        else:
            raise ValueError("Invalid image_size. It should be a tuple (width, height) or integer.")

        # Darknet 53
        # Stage 0
        x = self.conv2D_block(input, 32)
        x = self.conv2D_block(x, 64, strides=2)
        x = self.res_unit(x, 64)

        # Stage 1
        x = self.conv2D_block(x, 128, strides=2)
        for _ in range(2):
            x = self.res_unit(x, 128)

        # Stage 2
        x = self.conv2D_block(x, 256, strides=2)
        for _ in range(8):
            x = self.res_unit(x, 256)
        skip_L = x

        # Stage 3
        x = self.conv2D_block(x, 512, strides=2)
        for _ in range(8):
            x = self.res_unit(x, 512)
        skip_M = x

        # Stage 4
        x = self.conv2D_block(x, 1024, strides=2)
        for _ in range(4):
            x = self.res_unit(x, 1024)

        # YOLO small
        x = self.conv2D_unit(x, 1024)
        yolo_S = self.scale_prediction(x, 1024, name="small")

        # YOLO medium
        x = self.conv2D_block(x, 256, kernel=1, upsampling=True)
        x = Concatenate()([x, skip_M])
        x = self.conv2D_unit(x, 512)
        yolo_M = self.scale_prediction(x, 512, name="medium")

        # YOLO large
        x = self.conv2D_block(x, 128, kernel=1, upsampling=True)
        x = Concatenate()([x, skip_L])
        x = self.conv2D_unit(x, 256)
        yolo_L = self.scale_prediction(x, 256, name="large")
        
        # Output
        model = Model(inputs=[input], outputs=[yolo_S, yolo_M, yolo_L], name=model_name)
        return model