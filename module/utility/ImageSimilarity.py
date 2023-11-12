import json
import random
import numpy as np
from .VectorCalculator import VectorCalculator

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