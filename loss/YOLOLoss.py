import tensorflow as tf

class YOLOLoss(tf.keras.losses.Loss):
    """
    Custom YOLOv3 Loss Function for object detection models.

    This loss function computes the YOLOv3 loss, which is used in training YOLO-based object detection models.

    Args:
        lambda_coord (float): Weight for the box coordinate loss.
        lambda_noobj (float): Weight for the no-object confidence loss.

    Attributes:
        lambda_coord (float): Weight for the box coordinate loss.
        lambda_noobj (float): Weight for the no-object confidence loss.
        bce (tf.keras.losses.BinaryCrossentropy): Binary cross-entropy loss function.
        mse (tf.keras.losses.MeanSquaredError): Mean squared error loss function.
        cce (tf.keras.losses.CategoricalCrossentropy): Categorical cross-entropy loss function.

    Methods:
        __decoder(tensor):
            Decode the input tensor into object confidence, box coordinates, and class probabilities.
        call(y_true, y_pred):
            Compute the YOLOv3 loss based on true and predicted values.
    
    """
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def __decoder(self, tensor): # Format (Object_confidence, x, y, width, height, class_prob1, class_prob2, ...)
        object_score = tensor[..., 0]
        box_coordinate = tensor[..., 1:5]
        class_prob = tensor[..., 5:]
        return object_score, box_coordinate, class_prob
    
    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        # Decode
        true_score, true_box, true_class = self.__decoder(y_true)
        pred_score, pred_box, pred_class = self.__decoder(y_pred)

        # Create numpy array mask for object and no-object
        mask_obj = true_score == 1
        mask_noobj = true_score == 0

        # Confidence loss
        obj_loss = self.bce(true_score[mask_obj], pred_score[mask_obj])
        no_obj_loss = self.bce(true_score[mask_noobj], pred_score[mask_noobj])
        confidence_loss = obj_loss + (self.lambda_noobj * no_obj_loss)

        # Box coordinates loss
        box_loss = self.mse(true_box[mask_obj], pred_box[mask_obj])
        box_loss *= self.lambda_coord

        # Classification loss
        class_loss = self.cce(true_class[mask_obj], pred_class[mask_obj])

        # Calculate the total YOLOv3 loss
        total_loss = confidence_loss + box_loss + class_loss

        return total_loss