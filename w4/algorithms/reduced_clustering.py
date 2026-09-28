import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, rand_score
from sklearn.decomposition import KernelPCA
from algorithms.evaluation_tools import calculate_jaccard_index
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.pca import pca_n_components
from algorithms.pca_plots import plot_comparison_varying_components


def reduced_clustering_kmeans(X, y, features, dataset_name, clustering_algorithm):
    if clustering_algorithm == "kmeans++":
        run_analysis_varying_components(X, y, features, dataset_name)

    elif clustering_algorithm == "optics":
        run_analysis_varying_components_optics(X, y, features, dataset_name)


def run_analysis_varying_components(X, y, features, dataset_name, max_iter=100, n_runs=5):
    results = {'Without PCA': [], 'Our PCA': [], 'KernelPCA': []}
    n_components_values = list(range(2, 7))
    k = 2

    for method in results.keys():
        for n_components in n_components_values:
            silhouette_scores = []
            calinski_scores = []
            rand_scores = []
            jaccard_scores = []

            if method == "Without PCA":
                data = X.to_numpy()
            elif method == "Our PCA":
                _, _, _, data, _ = pca_n_components(X, features=features, view_plots=False, n_components=n_components)
            elif method == "KernelPCA":
                kpca = KernelPCA(n_components=n_components)
                data = kpca.fit_transform(X)

            for _ in range(n_runs):
                # Clustering con KMeans++
                kmeans = KMeansPlusPlus(k=k, distance_metric='manhattan', max_iter=max_iter)
                kmeans.fit_predict(data)

                # Compute labels
                labels_pred = np.zeros(len(data))
                for cluster_idx, indices in kmeans.clusters.items():
                    for i in indices:
                        labels_pred[i] = cluster_idx

                # Compute metrics
                sil = silhouette_score(data, labels_pred, metric='manhattan')
                silhouette_scores.append(sil)

                calinski = calinski_harabasz_score(data, labels_pred)
                calinski_scores.append(calinski)

                rand = rand_score(y, labels_pred)
                rand_scores.append(rand)

                jaccard = calculate_jaccard_index(labels_pred, y)
                jaccard_scores.append(jaccard)

            mean_sil = np.mean(silhouette_scores)
            mean_calinski = np.mean(calinski_scores)
            mean_rand = np.mean(rand_scores)
            mean_jaccard = np.mean(jaccard_scores)
            results[method].append([mean_sil, mean_calinski, mean_rand, mean_jaccard])

    # Convert results to NumPy arrays for easy plotting
    for method in results:
        results[method] = np.array(results[method])

    plot_comparison_varying_components(results, n_components_values, dataset_name, clustering_method="K-means++")


def run_analysis_varying_components_optics(X, y, features, dataset_name, n_runs=5):
    results = {'Without PCA': [], 'Our PCA': [], 'KernelPCA': []}
    n_components_values = list(range(2, 7))

    for method in results.keys():
        for n_components in n_components_values:
            silhouette_scores = []
            calinski_scores = []
            rand_scores = []
            jaccard_scores = []

            if method == "Without PCA":
                data = X.to_numpy()
            elif method == "Our PCA":
                _, _, _, data, _ = pca_n_components(X, features=features, view_plots=False, n_components=n_components)
            elif method == "KernelPCA":
                kpca = KernelPCA(n_components=n_components)
                data = kpca.fit_transform(X)

            for _ in range(n_runs):
                # Clustering with OPTICS
                optics = OPTICS(min_samples=5, metric='l1', xi=0.05)
                labels_pred = optics.fit_predict(data)

                # Handle unassigned points (-1 in OPTICS output)
                if -1 in labels_pred:
                    noise_indices = (labels_pred == -1)
                    labels_pred[noise_indices] = np.max(labels_pred) + 1  # Treat noise as a separate cluster

                # Compute metrics
                sil = silhouette_score(data, labels_pred, metric='manhattan')
                silhouette_scores.append(sil)

                calinski = calinski_harabasz_score(data, labels_pred)
                calinski_scores.append(calinski)

                rand = rand_score(y, labels_pred)
                rand_scores.append(rand)

                jaccard = calculate_jaccard_index(labels_pred, y)
                jaccard_scores.append(jaccard)

            # Store average metrics for this n_components
            mean_sil = np.mean(silhouette_scores)
            mean_calinski = np.mean(calinski_scores)
            mean_rand = np.mean(rand_scores)
            mean_jaccard = np.mean(jaccard_scores)
            results[method].append([mean_sil, mean_calinski, mean_rand, mean_jaccard])

    # Convert results to NumPy arrays for easy plotting
    for method in results:
        results[method] = np.array(results[method])

    plot_comparison_varying_components(results, n_components_values, dataset_name, clustering_method="Optics")


def input_kmeans_optics(df, y, features, dataset_name):
    while True:
        clustering_algorithm = input("Type the clustering algorithm ('kmeans++' or 'optics') or 'Exit' to go back to the menu: ")
        if clustering_algorithm not in ["kmeans++", "optics", "Exit"]:
            print("Invalid input format. Please try again.")
        elif clustering_algorithm == "Exit":
            break
        else:
            reduced_clustering_kmeans(df, y, features, dataset_name, clustering_algorithm)