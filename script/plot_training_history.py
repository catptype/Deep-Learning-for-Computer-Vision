import sys
sys.dont_write_bytecode = True

from matplotlib import pyplot as plt

def plot_training_history(history):
    """
    Plot the training history (accuracy and loss) of a machine learning model.

    Args:
        history (tf.keras.callbacks.History): The training history object obtained from model training.

    Example:
        To plot the training history of a TensorFlow model after training, you can use the following code:

        ```python
        history = model.fit(train_data, epochs=10, validation_data=val_data)
        plot_model_history(history)
        ```

    Note:
        This function assumes that the training history contains the following keys: 'accuracy', 'val_accuracy',
        'loss', and 'val_loss'. It plots both accuracy and loss curves for the training and validation datasets.
    """
    plt.figure(figsize=(20, 6))
    
    # Summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.show()
