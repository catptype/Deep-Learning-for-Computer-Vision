import numpy as np
from sklearn.cluster import KMeans

class YOLOv3Anchor:
    """
    Utility class for YOLOv3 anchor box calculations.

    Methods:
        find_best_anchor(width, height, anchor_list): Finds the best anchor box index for a given bounding box size.
        calculate_anchor(num_anchor, annotation_list): Calculates anchor boxes using k-means clustering based on annotation bounding box sizes.

    Note: This class provides methods for finding the best anchor box for a given bounding box size and
    calculating anchor boxes using k-means clustering based on annotation bounding box sizes.
    """

    @staticmethod
    def find_best_anchor(width, height, anchor_list):
        best_iou = 0
        best_anchor = None

        for idx, anchor in enumerate(anchor_list):
            anchor_width, anchor_height = anchor
            intersection = min(width, anchor_width) * min(height, anchor_height)
            union = width * height + anchor_width * anchor_height - intersection
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_anchor = idx

        return best_anchor

    @staticmethod
    def calculate_anchor(num_anchor, annotation_list):

        # Calculate n_anchor boudary box sizes (width, height) from all annotations
        bbox_size_list = []
        
        for annotation in annotation_list:
            for _, xmin, ymin, xmax, ymax in annotation:
                width = xmax - xmin
                height = ymax - ymin
                bbox_size_list.append([width, height])
        
        bbox_size_list = np.array(bbox_size_list)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_anchor, random_state=0)
        kmeans.fit(bbox_size_list)

        # Get the cluster centroids, which represent the anchor box sizes
        anchor_boxes = kmeans.cluster_centers_

        # Convert the NumPy array to a list of tuples
        anchor_boxes = [tuple(map(lambda x: round(x, 2), row)) for row in anchor_boxes]

        return anchor_boxes