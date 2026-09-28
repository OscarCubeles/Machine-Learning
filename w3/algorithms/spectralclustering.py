import numpy as np
import pandas as pd
from pyclustering.utils.metric import gower_distance
from tabulate import tabulate
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
from algorithms.evaluation_tools import calculate_modularity, plot_metrics, calculate_external_indices, calculate_dunn_score, gower_distance, minkowski_mixed_distance


class SpectralClusteringAlgorithm:
    def __init__(self, k, affinity, n_neighbors, eigen_solvers, assign_labels):
        self.k = k
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.eigen_solvers = eigen_solvers
        self.assign_labels = assign_labels
        self.affinity_matrix = None
        self.clusters = None

    def fit_predict(self, X):
        spectral_model = SpectralClustering(
            n_clusters=self.k,
            n_neighbors=self.n_neighbors,
            affinity=self.affinity,
            eigen_solver=self.eigen_solvers,
            assign_labels=self.assign_labels
        )

        spectral_model.fit(X)
        # Predict cluster labels
        labels_pred = spectral_model.labels_
        # Save affinity matrix
        self.affinity_matrix = spectral_model.affinity_matrix_
        # Get clusters
        clusters_dict = {}
        for cluster_label in np.unique(labels_pred):
            # Skip noise points
            if cluster_label != -1:
                cluster_indices = np.where(labels_pred == cluster_label)[0]
                clusters_dict[cluster_label] = cluster_indices
        self.clusters = clusters_dict

        return labels_pred


def run_test(X, dataset_name, k_values, numerical_columns):
    eigen_solvers = ['arpack', 'lobpcg']
    assign_labels = ['kmeans', 'cluster_qr']
    results = {}
    for k in tqdm(k_values):
        silhouette_scores = []
        modularity_scores = []
        dunn_scores = []
        dav_bouldin_scores = []
        for ass_lbl in assign_labels:
            for eigen_solver in eigen_solvers:
                spectral_cl = SpectralClusteringAlgorithm(k=k,
                                              affinity='rbf',
                                              n_neighbors=2,
                                              eigen_solvers=eigen_solver,
                                              assign_labels=ass_lbl)
                # Compute labels
                labels_pred = spectral_cl.fit_predict(X)

                # Compute Silhouette, Modularity and Davies Bouldin scores
                sil = silhouette_score(X, labels_pred, metric=lambda x, y: gower_distance(X, x, y, numerical_columns))
                silhouette_scores.append(sil)
                davb = davies_bouldin_score(X, labels_pred)
                dav_bouldin_scores.append(davb)
                dunn = calculate_dunn_score(X, labels_pred, numerical_columns)
                dunn_scores.append(dunn)
                mod = calculate_modularity(spectral_cl.affinity_matrix, labels_pred, X, True)
                modularity_scores.append(mod)

        mean_sil = np.mean(silhouette_scores)
        mean_mod = np.mean(modularity_scores)
        mean_dunn = np.mean(dunn_scores)
        mean_davb = np.mean(dav_bouldin_scores)
        metrics = np.array([mean_sil, mean_dunn, mean_davb, mean_mod])

        # Save metrics
        param_comb = 'rbf'
        if k == k_values[0]:
            results[param_comb] = metrics.reshape(-1, 1)
        else:
            results[param_comb] = np.hstack([results[param_comb], metrics.reshape(-1, 1)])

    metric_titles = ['Silhouette', 'Dunn', 'Davies-Bouldin', 'Modularity']
    plot_metrics(results, len(metrics), metric_titles, k_values, {v: k for k, v in {'orange': 'rbf'}.items()},
                 {'orange': 'rbf'}, dataset_name)

def run_internal_analysis(X, dataset_name, k_values, numerical_columns):
    affinities = ['nearest_neighbors', 'rbf']
    n_neighbors = [3, 7, 11]
    eigen_solvers = ['arpack', 'lobpcg']
    assign_labels = ['kmeans', 'cluster_qr']

    results = {}
    print("Analysing different methods for affinity computation... ")
    for aff in affinities:
        print(f'\n\tAffinity: {aff}')
        for n_neigh in n_neighbors:
            if aff != 'rbf':
                print(f'\n\t\tN neigh: {n_neigh}')
            for k in tqdm(k_values):
                silhouette_scores = []
                modularity_scores = []
                dav_bouldin_scores = []
                dunn_scores = []
                for ass_lbl in assign_labels:
                    for eigen_solver in eigen_solvers:
                        spectral_cl = SpectralClusteringAlgorithm(k=k,
                                                                  n_neighbors=n_neigh,
                                                                  affinity=aff,
                                                                  eigen_solvers=eigen_solver,
                                                                  assign_labels=ass_lbl)
                        # Compute labels
                        labels_pred = spectral_cl.fit_predict(X)

                        # Compute Silhouette, Modularity and Davies Bouldin scores
                        sil = silhouette_score(X, labels_pred)
                        silhouette_scores.append(sil)
                        davb = davies_bouldin_score(X, labels_pred)
                        dav_bouldin_scores.append(davb)
                        dunn = calculate_dunn_score(X, labels_pred, numerical_columns)
                        dunn_scores.append(dunn)
                        if aff == 'rbf':
                            mod = calculate_modularity(spectral_cl.affinity_matrix, labels_pred, X, True)
                        else:
                            mod = calculate_modularity(spectral_cl.affinity_matrix, labels_pred)
                        modularity_scores.append(mod)

                mean_sil = np.mean(silhouette_scores)
                mean_davb = np.mean(dav_bouldin_scores)
                mean_mod = np.mean(modularity_scores)
                mean_dunn = np.mean(dunn_scores)
                metrics = np.array([mean_sil, mean_mod, mean_dunn])

                # Save metrics
                if aff == 'nearest_neighbors':
                    param_comb = f'{aff}, k={n_neigh}'
                else:
                    param_comb = 'rbf'
                if k == k_values[0]:
                    results[param_comb] = metrics.reshape(-1, 1)
                else:
                    results[param_comb] = np.hstack([results[param_comb], metrics.reshape(-1, 1)])

            if aff == 'rbf': break

    # Plot metrics
    colors_labels_map = {'orange': 'rbf',
                         'lightblue': 'nearest_neighbors, k=3',
                         'dodgerblue': 'nearest_neighbors, k=7',
                         'darkblue': 'nearest_neighbors, k=11'}
    metric_titles = ['Silhouette', 'Modularity', 'Dunn']
    plot_metrics(results, len(metrics), metric_titles, k_values, {v: k for k, v in colors_labels_map.items()},
                 colors_labels_map, dataset_name, True)

    results = {}
    print("For affinity=rbf, analysing impact of eigen solver and label assignment method...")
    for ass_lbl in tqdm(assign_labels):
        print(f'Assign labels: {ass_lbl}')
        for eigen_solver in tqdm(eigen_solvers):
            print(f'Eigen solver: {eigen_solver}')
            for k in tqdm(k_values):
                spectral_cl = SpectralClusteringAlgorithm(k=k,
                                                          n_neighbors=11,
                                                          affinity='rbf',
                                                          eigen_solvers=eigen_solver,
                                                          assign_labels=ass_lbl)
                # Compute labels
                labels_pred = spectral_cl.fit_predict(X)

                # Compute Silhouette, Modularity and Davies Bouldin scores
                sil = silhouette_score(X, labels_pred)
                davb = davies_bouldin_score(X, labels_pred)
                mod = calculate_modularity(spectral_cl.affinity_matrix, labels_pred, X, True)
                dunn = calculate_dunn_score(X, labels_pred, numerical_columns)
                metrics = np.array([sil, mod, dunn])

                # Save metrics
                param_comb = f'{eigen_solver}, {ass_lbl}'
                if k == k_values[0]:
                    results[param_comb] = metrics.reshape(-1, 1)
                else:
                    results[param_comb] = np.hstack([results[param_comb], metrics.reshape(-1, 1)])

    # Plot metrics
    colors_labels_map = {'#FFA500': 'arpack, kmeans',
                         '#DAA520': 'arpack, cluster_qr',
                         '#FF8C00': 'lobpcg, kmeans',
                         '#FFD700': 'lobpcg, cluster_qr'}
    metric_titles = ['Silhouette', 'Modularity', 'Dunn']
    plot_metrics(results, len(metrics), metric_titles, k_values, {v: k for k, v in colors_labels_map.items()},
                 colors_labels_map, dataset_name+'2', True)


def run_external_analysis(X, y):
    results = pd.DataFrame(columns=['Purity', 'Adjusted Rand Index', 'Rand Index', 'Jaccard Index'])
    model1 = SpectralClusteringAlgorithm(k=2,
                                affinity='rbf',
                                n_neighbors=3,
                                eigen_solvers='arpack',
                                assign_labels='kmeans')
    labels1 = model1.fit_predict(X)
    results.loc[len(results)] = calculate_external_indices(labels1, y)
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))


def get_spectral_input():
    valid_affinities = ['nearest_neighbors', 'rbf']
    valid_eigen_solvers = ['arpack', 'lobpcg']
    valid_assign_labels = ['kmeans', 'cluster_qr']
    while True:
        print("\nPlease enter your command in the following format:")
        print("<k>-<affinity>-<eigen_solver>-<assign_labels> (e.g., 3-nearest_neighbors-arpack-kmeans)")
        user_input = input("Enter your command: ").strip()
        try:
            k_str, affinity, eigen_solver, assign_labels = user_input.split('-')
            k = int(k_str)

            if affinity not in valid_affinities:
                print(f"Error: Invalid affinity. Choose from {valid_affinities}.")
                continue
            elif affinity == 'nearest_neighbors':
                n_neighbors = int(input("Please enter number of neighbors to use when constructing the affinity matrix:"))
            else:
                n_neighbors = 10

            if eigen_solver not in valid_eigen_solvers:
                print(f"Error: Invalid eigen solver method. Choose from {valid_eigen_solvers}.")
                continue

            if assign_labels not in valid_assign_labels:
                print(f"Error: Invalid method for assigning labels. Choose from {valid_assign_labels}.")
                continue

            return k, affinity, n_neighbors, eigen_solver, assign_labels
        except ValueError:
            print("Error: Invalid input format. Please use the format <k>-<affinity>-<eigen_solver>-<assign_labels>.")
