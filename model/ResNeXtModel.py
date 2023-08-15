import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from .DeepLearningModel import DeepLearningModel

sys.dont_write_bytecode = True


class ResNeXtModel(DeepLearningModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def Conv2D_block(self, input, num_feature, kernel=3, strides=1, use_skip=False, identity=None):
        x = Conv2D(num_feature, (kernel, kernel), strides=strides, padding="same", kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        if use_skip:
            if x.shape[-1] != identity.shape[-1]:
                identity = self.Conv2D_block(identity, x.shape[-1], kernel=1)
            x = Add()([x, identity])
        x = ReLU()(x)
        return x

    def Resnext_block(self, input, num_feature, cardinality=32, downsampler=False):
        identity = input
        if downsampler:
            identity = MaxPooling2D()(identity)

        # Calculate the number of filters for each group
        feature_per_group = num_feature // cardinality

        group = []

        for _ in range(cardinality):
            if downsampler:
                x = self.Conv2D_block(input, feature_per_group, strides=2)
            else:
                x = self.Conv2D_block(input, feature_per_group)
            x = self.Conv2D_block(x, num_feature // 2)
            group.append(x)

        x = Add()(group)
        if x.shape[-1] != identity.shape[-1]:
            identity = self.Conv2D_block(identity, x.shape[-1], kernel=1)
        x = Add()([x, identity])

        return x

    def Resnext_bottleneck(self, input, num_feature, cardinality=32, downsampler=False):
        # Calculate the number of filters for each group
        feature_per_group = num_feature // cardinality

        group = []
        for _ in range(cardinality):
            x = self.Conv2D_block(input, feature_per_group, kernel=1)
            if downsampler:
                x = self.Conv2D_block(x, feature_per_group, kernel=3, strides=2)
            else:
                x = self.Conv2D_block(x, feature_per_group, kernel=3)
            group.append(x)

        x = Concatenate()(group)

        if downsampler:
            x = self.Conv2D_block(
                x,
                num_feature * 2,
                kernel=1,
                use_skip=True,
                identity=MaxPooling2D()(input),
            )
        else:
            x = self.Conv2D_block(
                x, num_feature * 2, kernel=1, use_skip=True, identity=input
            )

        return x


class ResNeXt18(ResNeXtModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        x = self.Resnext_block(x, 128)
        x = self.Resnext_block(x, 128)

        # stage 2
        x = self.Resnext_block(x, 256, downsampler=True)
        x = self.Resnext_block(x, 256)

        # stage 3
        x = self.Resnext_block(x, 512, downsampler=True)
        x = self.Resnext_block(x, 512)

        # stage 4
        x = self.Resnext_block(x, 1024, downsampler=True)
        x = self.Resnext_block(x, 1024)

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"ResNeXt18_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class ResNeXt34(ResNeXtModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Resnext_block(x, 128)

        # stage 2
        for i in range(4):
            x = (
                self.Resnext_block(x, 256, downsampler=True)
                if i == 0
                else self.Resnext_block(x, 256)
            )

        # stage 3
        for i in range(6):
            x = (
                self.Resnext_block(x, 512, downsampler=True)
                if i == 0
                else self.Resnext_block(x, 512)
            )

        # stage 4
        for i in range(3):
            x = (
                self.Resnext_block(x, 1024, downsampler=True)
                if i == 0
                else self.Resnext_block(x, 1024)
            )

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"ResNeXt34_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class ResNeXt50(ResNeXtModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Resnext_bottleneck(x, 128)

        # stage 2
        for i in range(4):
            x = (
                self.Resnext_bottleneck(x, 256, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 256)
            )

        # stage 3
        for i in range(6):
            x = (
                self.Resnext_bottleneck(x, 512, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 512)
            )

        # stage 4
        for i in range(3):
            x = (
                self.Resnext_bottleneck(x, 1024, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 1024)
            )

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"ResNeXt50_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class ResNeXt101(ResNeXtModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Resnext_bottleneck(x, 128)

        # stage 2
        for i in range(4):
            x = (
                self.Resnext_bottleneck(x, 256, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 256)
            )

        # stage 3
        for i in range(23):
            x = (
                self.Resnext_bottleneck(x, 512, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 512)
            )

        # stage 4
        for i in range(3):
            x = (
                self.Resnext_bottleneck(x, 1024, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 1024)
            )

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"ResNeXt101_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model


class ResNeXt152(ResNeXtModel):
    def __init__(self, image_size, num_classes):
        super().__init__(image_size=image_size, num_classes=num_classes)

    def build_model(self):
        # Input layer
        input = Input(shape=(self.image_size, self.image_size, 3), name="Input_image")

        # stage 0
        x = self.Conv2D_block(input, 64, kernel=7, strides=2)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # state 1
        for _ in range(3):
            x = self.Resnext_bottleneck(x, 128)

        # stage 2
        for i in range(8):
            x = (
                self.Resnext_bottleneck(x, 256, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 256)
            )

        # stage 3
        for i in range(36):
            x = (
                self.Resnext_bottleneck(x, 512, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 512)
            )

        # stage 4
        for i in range(3):
            x = (
                self.Resnext_bottleneck(x, 1024, downsampler=True)
                if i == 0
                else self.Resnext_bottleneck(x, 1024)
            )

        # output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation="softmax", dtype=tf.float32)(x)

        model = Model(
            inputs=[input],
            outputs=output,
            name=f"ResNeXt152_{self.image_size}x{self.image_size}_{self.num_classes}Class",
        )
        return model
