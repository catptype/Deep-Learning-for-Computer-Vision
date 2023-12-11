import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import mixed_precision

class DeepLearningModel(ABC):
    """
    Abstract base class for deep learning models.

    Provides common methods for building, compiling, training, and evaluating models.

    Attributes:
        model (tf.keras.Model): The constructed deep learning model.
        history (tf.keras.callbacks.History): History of training metrics.

    Methods:
        __init__(): Initialize the DeepLearningModel instance.
        build_model(): Abstract method to be implemented by subclasses for building the model.
        compile(optimizer, loss, metrics): Compile the model with optimizer, loss, and metrics.
        evaluate(test_data, test_labels): Evaluate the model on test data.
        get_input_shape(): Get the input shape of the model.
        get_model_name(): Get the name of the model.
        predict(data): Make predictions using the model.
        save(name=None): Save the model to a file.
        summary(): Display the summary of the model architecture.
        train(train_data, test_data=None, epochs=10, callbacks=None): Train the model on the given data.

    Note: Subclasses must implement the abstract method `build_model`.
    """
    def __init__(self):
        self.model = self.build_model()
        self.history = None

    @abstractmethod
    def build_model(self):
        pass

    def compile(
        self, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    ):
        if not isinstance(metrics, list):
            raise ValueError("metrics must be list.")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

    def get_input_shape(self):
        _, height, width, channel = self.model.input_shape
        return (height, width, channel)

    def get_model_name(self):
        return self.model.name
    
    def predict(self, data):
        return self.model.predict(data)
    
    def save(self, name=None):
        if name is None:
            model_name = self.model.name
        else:
            model_name = name
            
        if "mixed_float16" in str(mixed_precision.global_policy()):
            model_name += "_fp16"
        self.model.save(f"export model\\{model_name}.h5", include_optimizer=False)
    
    def summary(self):
        self.model.summary()

    def train(self, train_data, test_data=None, epochs=10, callbacks=None):
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