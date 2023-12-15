import json
import random
import numpy as np
from numpy.linalg import norm

class ImageSimilarity:
    """
    Utility class for retrieving similar images based on feature vectors.

    Methods:
        random_query_image(json_file): Selects a random image from the provided JSON file and returns its index and path.
        get_similar_image(json_file, query_idx): Retrieves a list of images sorted by similarity to the query image.

    Example:
        ```python
        # Example usage of ImageSimilarity class
        query_idx, query_path = ImageSimilarity.random_query_image('image_data.json')
        similar_images = ImageSimilarity.get_similar_image('image_data.json', query_idx)
        ```
    """
    @staticmethod
    def random_query_image(json_file):
        with open(json_file, 'r') as j:
            data = json.load(j)
        
        image_path_list = [entry['path'] for entry in data]
        query_idx = random.randrange(len(image_path_list) + 1)
        query_path = image_path_list[query_idx]
        return query_idx, query_path

    @staticmethod    
    def get_similar_image(json_file, query_idx):
        try:
            with open(json_file, 'r') as j:
                data = json.load(j)

            # Extract image paths and feature vectors
            image_path_list = [entry['path'] for entry in data]
            feature_vector_list = [np.array(entry['feature_vector']) for entry in data]

            query_vector = feature_vector_list[query_idx]
        
            cosine_list = VectorCalculator.cosine_similarity(query_vector, feature_vector_list)
            distance_list = VectorCalculator.euclidean_distance(query_vector, feature_vector_list)

            image_list = [(image_path, score, distance) for image_path, score, distance in zip(image_path_list, cosine_list, distance_list)]

            return image_list
        
        except:
            print(f"Your JSON file contains '{len(image_path_list)}' images.")
            return []

class VectorCalculator:
    """
    Utility class for calculating vector-based metrics.

    Methods:
        euclidean_distance(query_vector, dataset_vector): Computes the Euclidean distance between a query vector and a dataset of vectors.
        cosine_similarity(query_vector, dataset_vector): Computes the cosine similarity between a query vector and a dataset of vectors.

    Example:
        ```python
        # Example usage of VectorCalculator class
        distances = VectorCalculator.euclidean_distance(query_vector, dataset_vectors)
        similarities = VectorCalculator.cosine_similarity(query_vector, dataset_vectors)
        ```

    """
    @staticmethod
    def euclidean_distance(query_vector, dataset_vector):
        if len(query_vector) != len(dataset_vector[0]):
            raise ValueError("Query vector and dataset vectors must have the same length")
    
        distances = norm(dataset_vector - query_vector, axis=1)
        return distances

    @staticmethod
    def cosine_similarity(query_vector, dataset_vector):
        if len(query_vector) != len(dataset_vector[0]):
            raise ValueError("Query vector and dataset vectors must have the same length")
        
        cosine_list = np.dot(dataset_vector, query_vector) / (norm(dataset_vector, axis=1) * norm(query_vector))
        return cosine_list