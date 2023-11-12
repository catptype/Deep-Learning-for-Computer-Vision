import numpy as np
from numpy.linalg import norm

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