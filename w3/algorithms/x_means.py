from algorithms.kmeans import KMeans
import numpy as np


class XMeans(KMeans):
    def __init__(self, k_min=2, k_max=10, distance_metric='euclidean', max_iter=100):
        super().__init__(k=k_min, distance_metric=distance_metric, max_iter=max_iter)
        self.k_min = k_min
        self.k_max = k_max

    def fit_predict(self, X):

        # Start with KMeans for k_min clusters
        centroids, clusters = super().fit_predict(X)

        while len(centroids) < self.k_max:
            new_centroids = []
            should_split = False

            # Check each cluster to decide whether to split
            for cluster_idx, indices in clusters.items():
                cluster_points = np.array([X[i] for i in indices])

                if len(cluster_points) <= 1:  # Cannot split single-point clusters
                    new_centroids.append(centroids[cluster_idx])
                    continue

                # Perform KMeans with k=2 on this cluster
                sub_kmeans = KMeans(k=2, distance_metric=self.distance_metric, max_iter=self.max_iter)
                sub_centroids, sub_clusters = sub_kmeans.fit_predict(cluster_points)

                # Calculate BIC for current and split clusters
                bic_current = self._calculate_bic(cluster_points, [centroids[cluster_idx]])
                bic_split = self._calculate_bic(cluster_points, sub_centroids, sub_clusters)

                if bic_split > bic_current:
                    # Split the cluster
                    new_centroids.extend(sub_centroids)
                    should_split = True
                else:
                    new_centroids.append(centroids[cluster_idx])

            if not should_split:
                break

            # Update centroids and reassign points
            self.centroids = np.array(new_centroids)
            clusters = self._assign_clusters(X)
            centroids = self.centroids

        self.final_centroids = centroids
        self.final_clusters = clusters

        return self.final_centroids, self.final_clusters

    def _calculate_bic(self, data, centroids, clusters=None):
        """Calculate Bayesian Information Criterion (BIC)."""
        if clusters is None:
            clusters = {0: list(range(len(data)))}
        k_values = len(centroids)
        n_samples, n_features = data.shape

        # Log-likelihood
        log_likelihood = 0
        for i, indices in clusters.items():
            cluster_points = np.array([data[j] for j in indices])
            if len(cluster_points) == 0:
                continue
            variance = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
            log_likelihood += -len(cluster_points) * n_features * 0.5 * np.log(variance)

        # return BIC score
        return -2 * log_likelihood + (n_features * k_values) * np.log(n_samples)


def get_xmeans_input():
    valid_distances = ['euclidean', 'manhattan', 'cosine']
    while True:
        print("\nSelect the configuration to see the final clustering with PCA.")
        print("Please enter your command in the following format:")
        print("<k_min>-<k_max>-<distance_metric> (e.g., 2-10-euclidean)")
        user_input = input("Enter your command: ").strip()
        try:
            k_min_str, k_max_str, distance = user_input.split('-')
            k_min = int(k_min_str)
            k_max = int(k_max_str)

            if k_min <= 0 or k_max <= 0:
                print("Error: k_min and k_max must be positive integers.")
                continue

            if k_min >= k_max:
                print("Error: k_min must be less than k_max.")
                continue

            if distance not in valid_distances:
                print(f"Error: Invalid distance metric. Choose from {valid_distances}.")
                continue

            return k_min, k_max, distance
        except ValueError:
            print("Error: Invalid input format. Please use the format <k_min>-<k_max>-<distance>.")
