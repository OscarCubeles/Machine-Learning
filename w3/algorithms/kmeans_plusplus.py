import numpy as np
from algorithms.kmeans import KMeans


class KMeansPlusPlus(KMeans):

    def __init__(self, k, distance_metric='euclidean', max_iter=100):
        super().__init__(k, distance_metric, max_iter)

    def _initialize_centroids(self, X):
        centroids = []
        # Randomly select the first centroid
        centroids.append(X[np.random.randint(X.shape[0])])

        # Select remaining k-1 centroids
        for _ in range(1, self.k):
            distances = []
            for point in X:
                # Compute the distance from the point to the nearest centroid
                dist_to_nearest = min([self._get_distance(point, c) for c in centroids])
                if np.isnan(dist_to_nearest):
                    distances.append(0.0001)
                else:
                    distances.append(dist_to_nearest ** 2)

            # Select the next centroid with a probability proportional to distance squared
            probabilities = distances / np.sum(distances)

            next_centroid_index = np.random.choice(range(X.shape[0]), p=probabilities)
            centroids.append(X[next_centroid_index])

        return np.array(centroids)

    def fit_predict(self, X):

        # Initialize centroids using K-Means++ method
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            self.clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return self.centroids, self.clusters



