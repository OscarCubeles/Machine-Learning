import numpy as np
import random


class KMeans:
    def __init__(self, k, distance_metric='euclidean', max_iter=100):
        self.k = k
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.centroids = None
        self.clusters = None
        self.labels = None

    def fit_predict(self, X):
        # Initialize centroids randomly from the data
        self.centroids = random.sample(list(X), self.k)
        self.centroids = np.array(self.centroids)

        for _ in range(self.max_iter):
            #  Assign clusters
            self.clusters = self._assign_clusters(X)
            #  Update centroids
            new_centroids = self._update_centroids(X)
            # If centroids don't change, the algorithm has converged
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids, self.clusters

    def _assign_clusters(self, data):

        clusters = {}

        for i, point in enumerate(data):
            distances = []
            for centroid in self.centroids:
                if self.distance_metric == 'euclidean':
                    distance = self._euclidean_distance(point, centroid)
                elif self.distance_metric == 'manhattan':
                    distance = self._manhattan_distance(point, centroid)
                elif self.distance_metric == 'cosine':
                    distance = self._cosine_similarity(point, centroid)
                distances.append(distance)

            # Assign point to the cluster with the closest centroid
            closest_centroid = np.argmin(distances)
            if closest_centroid not in clusters:
                clusters[closest_centroid] = []
            clusters[closest_centroid].append(i)

        return clusters

    def _update_centroids(self, data):
        new_centroids = []

        for cluster_idx in range(self.k):
            if cluster_idx in self.clusters and len(self.clusters[cluster_idx]) > 0:
                # If there are points in the cluster, compute the mean centroid
                cluster_points = [data[i] for i in self.clusters[cluster_idx]]
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                # If no points in the cluster, retain the old centroid
                new_centroid = self.centroids[cluster_idx]
            new_centroids.append(new_centroid)

        return np.array(new_centroids)

    def _get_distance(self, point1, point2):
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(point1, point2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(point1, point2)
        elif self.distance_metric == 'cosine':
            return self._cosine_similarity(point1, point2)

    # Distance Functions
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))

    def _cosine_similarity(self, point1, point2):
        dot_product = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1 - (dot_product / (norm1 * norm2))

    def compute_labels(self, X):
        # Compute labels
        labels_pred = np.zeros(len(X))
        for cluster_idx, indices in self.clusters.items():
            for i in indices:
                labels_pred[i] = cluster_idx
        self.labels = labels_pred


def get_kmeans_input():
    """
    Prompt the user to enter the value of k and the distance metric in the format <k>-<distance>.
    Validates the input and returns the parsed k and distance metric.
    """
    valid_distances = ['euclidean', 'manhattan', 'cosine']
    while True:
        print("\nSelect the configuration to see the final clustering with PCA.")
        print("Please enter your command in the following format:")
        print("<k>-<distance_metric> (e.g., 2-euclidean)")
        user_input = input("Enter your command: ").strip()
        try:
            k_str, distance = user_input.split('-')
            k = int(k_str)

            if k <= 0 or k > 10:
                print("Error: k must be an integer between 1 and 10.")
                continue

            if distance not in valid_distances:
                print(f"Error: Invalid distance metric. Choose from {valid_distances}.")
                continue

            return k, distance
        except ValueError:
            print("Error: Invalid input format. Please use the format <k>-<distance>.")


def get_kmeans_distance():
    """
    Prompt the user to enter the distance metric in the format <distance>.
    Validates the input and returns the parsed distance metric.
    """
    valid_distances = ['euclidean', 'manhattan', 'cosine']
    while True:
        print("\nPlease enter your command in the following format:")
        print("<distance_metric> (e.g., euclidean)")
        user_input = input("Enter your command: ")
        try:
            distance = user_input

            if distance not in valid_distances:
                print(f"Error: Invalid distance metric. Choose from {valid_distances}.")
                continue

            return distance
        except ValueError:
            print("Error: Invalid input format. Please use the format <distance>.")
