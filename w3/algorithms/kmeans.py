import numpy as np
import random
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, f1_score, rand_score
from algorithms.evaluation_tools import plot_metrics, calculate_external_indices


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


def run_internal_analysis(X, y, dataset_name, k_values, max_iter=100, n_runs=3):
    results = {'euclidean': [], 'manhattan': [], 'cosine': []}

    for distance_metric in tqdm(results.keys()):
        for k in k_values:
            silhouette_scores = []
            calinski_scores = []
            f1_scores = []
            rand_scores = []
            for _ in range(n_runs):

                kmeans = KMeans(k=k, distance_metric=distance_metric, max_iter=max_iter)
                kmeans.fit_predict(X)

                # Compute labels
                labels_pred = np.zeros(len(X))
                for cluster_idx, indices in kmeans.clusters.items():
                    for i in indices:
                        labels_pred[i] = cluster_idx

                # Compute Silhouette and Distorsion
                sil = silhouette_score(X, labels_pred, metric=distance_metric)
                silhouette_scores.append(sil)
                # Calisnki
                calinski = calinski_harabasz_score(X, labels_pred)
                calinski_scores.append(calinski)
                # F1 score
                f1 = f1_score(y, labels_pred, average='weighted')
                f1_scores.append(f1)
                # Rand score
                rand = rand_score(y, labels_pred)
                rand_scores.append(rand)

            # Compute mean metrics through runs and save them
            mean_sil = np.mean(silhouette_scores)
            mean_calinski = np.mean(calinski_scores)
            mean_f1 = np.mean(f1_scores)
            mean_rand = np.mean(rand_scores)
            metrics = np.array([mean_sil, mean_calinski, mean_rand])
            if k == k_values[0]:
                results[distance_metric] = metrics.reshape(-1, 1)
            else:
                results[distance_metric] = np.hstack([results[distance_metric], metrics.reshape(-1, 1)])

    # Plot metrics
    colors_labels_map = {'orange': 'euclidean', 'green': 'manhattan', 'blue': 'cosine'}
    metric_titles = ['Silhouette', 'Calinski', 'Rand score']
    plot_metrics(results, len(metrics), metric_titles, k_values, {v: k for k, v in colors_labels_map.items()},
                 colors_labels_map, dataset_name)


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


def run_external_analysis(X, y, dataset_name):
    # Define k values based on the dataset
    if dataset_name == "hepatitis":
        k1, k2, k3 = 2, 3, 6
    elif dataset_name == "mx":
        k1, k2, k3 = 2, 4, 6
    elif dataset_name == "breast":
        k1, k2, k3 = 2, 3, 5

    n_runs = 5
    all_labels_m1 = np.zeros((len(X), n_runs), dtype=int)
    all_labels_m2 = np.zeros((len(X), n_runs), dtype=int)
    all_labels_m3 = np.zeros((len(X), n_runs), dtype=int)
    for run_idx in range(n_runs):
        # Initialize and fit the KMeans models
        kmeans1 = KMeans(k=k1, distance_metric="cosine")
        kmeans1.fit_predict(X)
        kmeans1.compute_labels(X)
        all_labels_m1[:, run_idx] = kmeans1.labels

        kmeans2 = KMeans(k=k2, distance_metric="cosine")
        kmeans2.fit_predict(X)
        kmeans2.compute_labels(X)
        all_labels_m2[:, run_idx] = kmeans2.labels

        kmeans3 = KMeans(k=k3, distance_metric="cosine")
        kmeans3.fit_predict(X)
        kmeans3.compute_labels(X)
        all_labels_m3[:, run_idx] = kmeans3.labels

    final_labels_m1 = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m1
    )
    final_labels_m2 = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m2
    )
    final_labels_m3 = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m3
    )

    # Create DataFrame to store results with additional columns for 'Distance' and 'k'
    results = pd.DataFrame(columns=['Distance', 'Purity', 'F1 score', 'Jaccard Index'],
                           index=[f'Model k={k1}', f'Model k={k2}', f'Model k={k3}'])

    # Calculate external indices and populate the table with results, distance, and k values
    results.loc[f'Model k={k1}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m1, y)
    results.loc[f'Model k={k1}', 'Distance'] = "Cosine"

    results.loc[f'Model k={k2}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m2, y)
    results.loc[f'Model k={k2}', 'Distance'] = "Cosine"

    results.loc[f'Model k={k3}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m3, y)
    results.loc[f'Model k={k3}', 'Distance'] = "Cosine"

    # Print the results in a fancy grid table format
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))