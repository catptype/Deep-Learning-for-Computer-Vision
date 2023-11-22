import json
import random
import numpy as np
from numpy.linalg import norm

class ImageSimilarity:

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