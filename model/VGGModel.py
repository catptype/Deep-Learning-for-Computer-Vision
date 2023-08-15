import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel

sys.dont_write_bytecode = True


class VGGModel(DeepLearningModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def Conv2D_block(self, input, num_feature, kernel=3, use_bn=False, downsampler=False):
        x = Conv2D(num_feature, (kernel, kernel), padding="same", kernel_initializer="he_normal")(input)
        if use_bn:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        if downsampler:
            x = MaxPooling2D(strides=2)(x)
        return x


class CustomVGG(VGGModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64, use_bn=True)
        x = self.Conv2D_block(x, 64, use_bn=True, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128, use_bn=True)
        x = self.Conv2D_block(x, 128, use_bn=True, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256, use_bn=True)
        x = self.Conv2D_block(x, 256, use_bn=True, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"CustomVGG_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class VGG11(VGGModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG11_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class VGG13(VGGModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(x, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128)
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG13_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class VGG16(VGGModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(x, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(x, 128)
        x = self.Conv2D_block(x, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256)
        x = self.Conv2D_block(x, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512)
        x = self.Conv2D_block(x, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG16_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class VGG19(VGGModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 1
        x = self.Conv2D_block(input, 64)
        x = self.Conv2D_block(input, 64, downsampler=True)

        # stage 2
        x = self.Conv2D_block(input, 128)
        x = self.Conv2D_block(input, 128, downsampler=True)

        # stage 3
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256)
        x = self.Conv2D_block(input, 256, downsampler=True)

        # stage 4
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512, downsampler=True)

        # stage 5
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512)
        x = self.Conv2D_block(input, 512, downsampler=True)

        # output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"VGG19_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model
