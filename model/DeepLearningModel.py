import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import mixed_precision

class DeepLearningModel(ABC):
    """
    Abstract base class for deep learning models.

    Attributes:
        model (tf.keras.Model): The neural network model.
        history (tf.keras.callbacks.History): Training history.

    Methods:
        build_model(): Abstract method to build the neural network architecture.
        summary(): Display a summary of the model's architecture.
        compile(optimizer, loss, metrics): Compile the model for training.
        train(train_data, test_data, epochs, callbacks): Train the model.
        evaluate(test_data, test_labels): Evaluate the model on test data.
        predict(data): Make predictions on input data.
        save(name): Save the trained model to a file.
    """
    def __init__(self):
        self.model = self.build_model()
        self.history = None

    @abstractmethod
    def build_model(self):
        """
        Abstract method to build the neural network architecture.
        Subclasses must implement this method.
        """
        pass

    def summary(self):
        """Display a summary of the model's architecture."""
        self.model.summary()

    def compile(self, 
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]):
        """
        Compile the model for training.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.
            loss (tf.keras.losses.Loss): The loss function.
            metrics (list): List of metrics to monitor during training.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_data, test_data=None, epochs=10, callbacks=None):
        """
        Train the model.

        Args:
            train_data: Training data.
            test_data: Validation data (optional).
            epochs (int): Number of training epochs.
            callbacks (list): List of callbacks for training.
        """
        if self.history is None:
            # Train the model from scratch and store the training history
            self.history = self.model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=callbacks)
        else:
            print("Continuing training from epoch", max(self.history.epoch) + 1)
            new_history = self.model.fit(train_data, initial_epoch=max(self.history.epoch)+1, epochs=epochs, validation_data=test_data, callbacks=callbacks)
            
            # Update the training history with the new epoch values and metrics
            self.history.epoch += [epoch for epoch in new_history.epoch]
            for key in self.history.history.keys():
                self.history.history[key] += new_history.history[key]

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data.
            test_labels: Ground truth labels for test data.

        Returns:
            List of evaluation metrics.
        """
        return self.model.evaluate(test_data, test_labels)

    def predict(self, data):
        """
        Make predictions on input data.

        Args:
            data: Input data for predictions.

        Returns:
            Model predictions.
        """
        return self.model.predict(data)

    def save(self, name=None):
        """
        Save the trained model to a file.

        Args:
            name (str): Optional. The name to use for the saved model file (without file extension).
                        If not provided, the model's name is used as the default name.

        Note:
            - The saved model file will have an ".h5" extension.
            - If mixed precision training (FP16) is enabled, "_fp16" is appended to the model name.
            - The saved model will be stored in the "export model" directory.
        """
        if name is None:
            model_name = self.model.name
        else:
            model_name = name
            
        if "mixed_float16" in str(mixed_precision.global_policy()):
            model_name += "_fp16"
        self.model.save(f"export model\\{model_name}.h5", include_optimizer=False)

