import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from algorithms.evaluation_tools import calculate_hamming_matrix, plot_hamming_matrices, calculate_connectivity, \
    calculate_cluster_density_kde, calculate_external_indices, plot_confusion_matrix_heatmap, gower_distance, pairwise_dist_kde, \
    plot_reachability_distances


class OpticsClustering:
    def __init__(self, metric, algorithm, min_samples, xi):
        self.metric = metric
        self.algorithm = algorithm
        self.min_samples = min_samples
        self.xi = xi
        self.optics_model = None
        self.clusters = None

    def fit_predict(self, X):
        optics_model = OPTICS(
            metric=self.metric,
            algorithm=self.algorithm,
            min_samples=self.min_samples,
            min_cluster_size = 0.03,
            xi=self.xi
        )
        optics_model.fit(X)
        self.optics_model = optics_model
        # Predict cluster labels
        labels_pred = optics_model.labels_
        # Get clusters
        clusters_dict = {}
        for cluster_label in np.unique(labels_pred):
            # Skip noise points
            if cluster_label != -1:
                cluster_indices = np.where(labels_pred == cluster_label)[0]
                clusters_dict[cluster_label] = cluster_indices
        self.clusters = clusters_dict

        return labels_pred


def run_internal_analysis(X, dataset_name, numerical_columns):
    # First plot the point density of the dataset
    pairwise_dist_kde(X, dataset_name, True)

    # Define parameter min_samples and xi based on pairwise distances distribution
    min_samples_xi_dict = {'hepatitis':{'min_samples':15, 'xi':0.001},
                        'mx':{'min_samples':5, 'xi':0.05},
                        'breast':{'min_samples':3, 'xi':0.5}}

    distances = ['euclidean', 'cosine', 'l1']
    algorithms = ['ball_tree', 'brute']

    results = []
    labels_dict = {}
    n_comb = 0
    print("Analysing different metrics...")
    for distance in tqdm(distances):
        print("Analysing different algorithms for nearest neighbors...")
        for algorithm in tqdm(algorithms):
            if distance == 'cosine' and algorithm == 'ball_tree':
                continue
            optics = OpticsClustering(metric=distance,
                                      algorithm=algorithm,
                                      min_samples=min_samples_xi_dict[dataset_name]['min_samples'],
                                      xi=min_samples_xi_dict[dataset_name]['xi'])

            # Compute labels
            labels_pred = optics.fit_predict(X)
            labels_dict[f"{n_comb}"] = labels_pred
            valid_labels = labels_pred != -1
            n_clusters = len(set(labels_pred[valid_labels]))

            # Compute Silhouette, Connectivity and Cluster density scores
            sil = silhouette_score(X[valid_labels], labels_pred[valid_labels], metric=distance) if n_clusters > 1 else -1
            conn = calculate_connectivity(X, labels_pred, distance, numerical_columns)
            cl_den = calculate_cluster_density_kde(X, labels_pred)

            # Save metrics
            results.append({'metric': distance,
                            'algorithm': algorithm,
                            'silhouette_score': sil,
                            'connectivity': conn,
                            'cluster_density': cl_den,
                            '#clusters': n_clusters,
                            '#noise_pts': len(labels_pred[labels_pred == -1])})
            n_comb += 1

    # Show results
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

    # Compute and plot hamming matrix
    hamming_matrix = calculate_hamming_matrix(labels_dict)
    plot_hamming_matrices(hamming_matrix, len(X), dataset_name=dataset_name, x_labels=['euclidean, ball_tree', 'euclidean, brute', 'cosine, brute', 'l1, ball_tree', 'l1, brute'], save=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'data/results/{dataset_name}_OPTICS', index=False)


def run_external_analysis(X, y, dataset_name):

    optics = OpticsClustering('cosine', 'brute', 15, 0.001)
    labels = optics.fit_predict(X)
    valid_labels = labels != -1

    results = pd.DataFrame(columns=['Purity', 'Adjusted Rand Index', 'Rand Index', 'Jaccard Index'])

    results.loc[len(results)] = calculate_external_indices(labels[valid_labels], y[valid_labels])

    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

    plot_reachability_distances(optics.optics_model.reachability_, optics.optics_model.ordering_,
                                labels, dataset_name, save=True)
