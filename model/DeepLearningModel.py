import sys
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import mixed_precision

sys.dont_write_bytecode = True


class DeepLearningModel(ABC):
    def __init__(self, image_size, num_classes):
        """
        image_size (int): The size of input images for the model.
        num_classes (int): The number of classes for classification tasks.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for model training.
        loss (tf.keras.losses.Loss): The loss function to use for model training.
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def summary(self):
        self.model.summary()

    def compile(self, 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_checkpoint(self, train_data, test_data=None, epochs=10, save_interval=5):
        if "mixed_float16" in str(mixed_precision.global_policy()):
            ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
            ckpt_dir = f"CHECKPOINT/{self.model.name}_fp16/"
        else:
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            ckpt_dir = f"CHECKPOINT/{self.model.name}/"
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

        latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)

        if latest_checkpoint is not None:
            print(f"Resuming training from checkpoint: {latest_checkpoint}\n")
            self.model.load_weights(latest_checkpoint)
            current_epoch = int(latest_checkpoint.split("-")[-1])
            initial_epoch = current_epoch * save_interval
        else:
            print("No checkpoint found")
            initial_epoch = 0

        while initial_epoch < epochs:
            process_epoch = initial_epoch + save_interval

            if process_epoch < epochs:
                self.model.fit(
                    train_data,
                    initial_epoch=initial_epoch,
                    epochs=process_epoch,
                    validation_data=test_data,
                )

                print(f"Saving checkpoint at Epoch {process_epoch} ... ", end="")
                ckpt_manager.save(checkpoint_number=process_epoch // save_interval)
                print("Done")

                initial_epoch += save_interval
            else:
                self.model.fit(
                    train_data,
                    initial_epoch=initial_epoch,
                    epochs=epochs,
                    validation_data=test_data,
                )
                break

    def train(self, train_data, test_data=None, epochs=10):
        EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     min_delta=0.01, 
                                                     patience=5, 
                                                     restore_best_weights=True, 
                                                     verbose=1)
        self.model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=[EarlyStop])

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

    def predict(self, data):
        return self.model.predict(data)

    def save(self):
        model_name = self.model.name
        if "mixed_float16" in str(mixed_precision.global_policy()):
            model_name += "_fp16"
        self.model.save(f"export model\\{model_name}.h5")
