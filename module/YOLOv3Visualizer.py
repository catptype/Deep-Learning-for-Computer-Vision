import sys
sys.dont_write_bytecode = True

import cv2
import matplotlib.pyplot as plt
import numpy as np

class YOLOv3Visualizer:
    def __init__(self, class_mapping=None):
        self.class_mapping = class_mapping
    
    # Private methods
    def __draw_grid(self, image, x_grid_size, y_grid_size):
        # Extract variables
        image_height, image_width = image.shape[:2]

        # Boundary box setup
        grid_color = (0, 0, 255)  # blue for grid lines
        grid_thickness = 1

        cell_width = image_width // x_grid_size
        cell_height = image_height // y_grid_size

        # Draw vertical grid lines
        for i in range(x_grid_size):
            x = i * cell_width
            cv2.line(image, (x, 0), (x, image.shape[0]), grid_color, grid_thickness)

        # Draw horizontal grid lines
        for i in range(y_grid_size):
            y = i * cell_height
            cv2.line(image, (0, y), (image.shape[1], y), grid_color, grid_thickness)

    def __draw_highlight(self, image, x_center, y_center, grid_info):
        # Extract variables
        image_height, image_width = image.shape[:2]
        row, col, x_grid_size, y_grid_size = grid_info

        # Boundary box setup
        grid_color = (255, 0, 0)  # Red for grid lines
        grid_thickness = 1

        # Convert grid cell coordinates to image coordinates
        x_center_image = (row + x_center) * image_width / x_grid_size
        y_center_image = (col + y_center) * image_height / y_grid_size
        cell_width = image_width // x_grid_size
        cell_height = image_height // y_grid_size
        
        x_min = row * cell_width
        y_min = col * cell_height
        x_max = (row + 1) * cell_width
        y_max = (col + 1) * cell_height
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), grid_color, grid_thickness)

        # Draw dots at x_center and y_center
        dot_radius = 2
        dot_color = (255, 0, 0)  # Blue for dots
        x_center_pixel = int(x_center_image)
        y_center_pixel = int(y_center_image)

        cv2.circle(image, (x_center_pixel, y_center_pixel), dot_radius, dot_color, -1)
        
    def __draw_bounding_box(self, image, bounding_box, grid_info):
        # Extract variables
        image_height, image_width = image.shape[:2]
        x_center, y_center, width, height, class_name, objectness = bounding_box
        row, col, x_grid_size, y_grid_size = grid_info

        # Boundary box setup
        color = (0, 255, 0)  # Green for the bounding box color
        thickness = 2

        # Convert grid cell coordinates to image coordinates
        x_center_image = (row + x_center) * image_width / x_grid_size
        y_center_image = (col + y_center) * image_height / y_grid_size
        width_image = width * image_width
        height_image = height * image_height

        # Calculate bounding box coordinates
        x_min = int(x_center_image - width_image / 2)
        y_min = int(y_center_image - height_image / 2)
        x_max = int(x_center_image + width_image / 2)
        y_max = int(y_center_image + height_image / 2)

        # Draw boundary box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(image, f"{class_name} ({objectness:.2f})", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Public method
    def display_batch(self, dataset, figsize=(10, 10), show_grid=False, show_highlight=False):
        plt.figure(figsize=figsize)

        for images, labels in dataset.take(1):
            batch_size = images.shape[0]
            num_scales = len(labels)

            for batch_idx in range(batch_size):
                image = (images[batch_idx] * 255).numpy().astype("uint8")

                for scale_idx, scale_labels in enumerate(labels):
                    plt.subplot(batch_size, num_scales, (batch_idx * num_scales) + (scale_idx + 1))
                    image_with_boxes = image.copy()

                    x_grid_size = scale_labels.shape[1]
                    y_grid_size = scale_labels.shape[2]
                    num_anchors = scale_labels.shape[3]

                    for row in range(x_grid_size):
                        for col in range(y_grid_size):
                            for anchor_idx in range(num_anchors):
                                objectness = scale_labels[batch_idx, row, col, anchor_idx, 0]

                                if objectness > 0:
                                    x_center, y_center, width, height = scale_labels[batch_idx, row, col, anchor_idx, 1:5]
                                    class_probs = scale_labels[batch_idx, row, col, anchor_idx, 5:]

                                    class_id = np.argmax(class_probs)
                                    class_name = self.class_mapping[class_id] if self.class_mapping else str(class_id)

                                    bounding_box = (x_center, y_center, width, height, class_name, objectness)
                                    grid_info = (row, col, x_grid_size, y_grid_size)

                                    self.__draw_bounding_box(image_with_boxes, bounding_box, grid_info)
                                    if show_grid:
                                        self.__draw_grid(image_with_boxes, x_grid_size, y_grid_size)
                                    if show_highlight:
                                        self.__draw_highlight(image_with_boxes, x_center, y_center, grid_info)

                    plt.imshow(image_with_boxes)
                    plt.title(f"s_{scale_idx}: g:{row+1}x{col+1} | c:({x_center:.2f}, {y_center:.2f})")
                    plt.axis("off")

        plt.show()