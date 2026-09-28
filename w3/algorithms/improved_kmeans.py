import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, f1_score, rand_score
from tabulate import tabulate
from tqdm import tqdm
from preprocessing.breastPreprocessor import BreastPreprocessor
from preprocessing.hepPreprocessor import HepPreprocessor
from preprocessing.mxPreprocessor import MxPreprocessor

from algorithms.kmeans import get_kmeans_input, KMeans
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.x_means import XMeans, get_xmeans_input
from algorithms.evaluation_tools import plot_clusters, plot_metrics, calculate_external_indices, \
    calculate_jaccard_index, calculate_purity


def kmeans_plus_plus(X, y, dataset_name, max_iter=300):
    run_internal_analysis_improved(X, y, "KMeans++", dataset_name, max_iter=100, n_runs=5)
    # run_external_analysis_improved(X, y, dataset_name, "KMeans++")
    k, distance_metric = get_kmeans_input()
    kmeans_pp = KMeansPlusPlus(k=k, distance_metric=distance_metric, max_iter=max_iter)
    kmeans_pp.fit_predict(X)
    plot_clusters(X, dataset_name, kmeans_pp.clusters, kmeans_pp.centroids)


def x_means(X, y, dataset_name, max_iter=100):
    run_internal_analysis_improved(X, y, "KMeans++", dataset_name, max_iter=100, n_runs=5)
    # run_external_analysis_improved(X, y, dataset_name, "x_means")
    k_min, k_max, distance_metric = get_xmeans_input()
    xmeans = XMeans(k_min=k_min, k_max=k_max, max_iter=max_iter, distance_metric=distance_metric)
    xmeans.fit_predict(X)
    plot_clusters(X, dataset_name, xmeans.final_clusters, xmeans.final_centroids)


def comparison_plot(k, n_runs=15, max_iter=100):
    preprocessor_hep = HepPreprocessor()
    preprocessor_mx = MxPreprocessor()
    preprocessor_breast = BreastPreprocessor()
    preprocessors = [preprocessor_hep, preprocessor_mx, preprocessor_breast]

    dataset_names = ["Hepatitis", "Mx", "Breast"]

    # Containers for scores per algorithm and dataset
    silhouette_scores = {name: {'KMeans': [], 'KMeans++': [], 'XMeans': []} for name in dataset_names}
    f1_scores = {name: {'KMeans': [], 'KMeans++': [], 'XMeans': []} for name in dataset_names}
    jaccard_scores = {name: {'KMeans': [], 'KMeans++': [], 'XMeans': []} for name in dataset_names}
    purity_scores = {name: {'KMeans': [], 'KMeans++': [], 'XMeans': []} for name in dataset_names}
    rand_scores = {name: {'KMeans': [], 'KMeans++': [], 'XMeans': []} for name in dataset_names}

    for preprocessor, dataset_name in zip(preprocessors, dataset_names):
        if dataset_name == "Mx":
            distance_metric = 'manhattan'
        else:
            distance_metric = 'cosine'
        X, y_true = preprocessor.X.to_numpy(), preprocessor.y.to_numpy()
        for run in range(n_runs):
            # Run KMeans
            kmeans = KMeans(k=k, distance_metric=distance_metric, max_iter=max_iter)
            kmeans.fit_predict(X)
            labels_km = np.zeros(len(X))
            for cluster_idx, indices in kmeans.clusters.items():
                for i in indices:
                    labels_km[i] = cluster_idx
            silhouette_scores[dataset_name]['KMeans'].append(silhouette_score(X, labels_km, metric=distance_metric))
            f1_scores[dataset_name]['KMeans'].append(f1_score(y_true, labels_km, average='weighted'))
            jaccard_scores[dataset_name]['KMeans'].append(calculate_jaccard_index(labels_km, y_true))
            purity_scores[dataset_name]['KMeans'].append(calculate_purity(y_true, labels_km))
            rand_scores[dataset_name]['KMeans'].append(rand_score(y_true, labels_km))

            # Run KMeans++
            kmeans_pp = KMeansPlusPlus(k=k, distance_metric=distance_metric, max_iter=max_iter)
            kmeans_pp.fit_predict(X)
            labels_kmpp = np.zeros(len(X))
            for cluster_idx, indices in kmeans_pp.clusters.items():
                for i in indices:
                    labels_kmpp[i] = cluster_idx
            silhouette_scores[dataset_name]['KMeans++'].append(silhouette_score(X, labels_kmpp, metric=distance_metric))
            f1_scores[dataset_name]['KMeans++'].append(f1_score(y_true, labels_kmpp, average='weighted'))
            jaccard_scores[dataset_name]['KMeans++'].append(calculate_jaccard_index(labels_kmpp, y_true))
            purity_scores[dataset_name]['KMeans++'].append(calculate_purity(y_true, labels_kmpp))
            rand_scores[dataset_name]['KMeans++'].append(rand_score(y_true, labels_kmpp))

            # Run X-Means
            xmeans = XMeans(k_min=k, max_iter=max_iter, distance_metric=distance_metric)
            xmeans.fit_predict(X)
            labels_xm = np.zeros(len(X))
            for cluster_idx, indices in xmeans.clusters.items():
                for i in indices:
                    labels_xm[i] = cluster_idx
            silhouette_scores[dataset_name]['XMeans'].append(silhouette_score(X, labels_xm, metric=distance_metric))
            f1_scores[dataset_name]['XMeans'].append(f1_score(y_true, labels_xm, average='weighted'))
            jaccard_scores[dataset_name]['XMeans'].append(calculate_jaccard_index(labels_xm, y_true))
            purity_scores[dataset_name]['XMeans'].append(calculate_purity(y_true, labels_xm))
            rand_scores[dataset_name]['XMeans'].append(rand_score(y_true, labels_kmpp))


    # Helper function to plot scores
    def plot_scores(scores, metric_name):
        plt.figure(figsize=(10, 6))
        bar_width = 0.15
        x_positions = np.arange(len(dataset_names))
        colors = ['#1f77b4', '#4c9ed9', '#6baed6']

        for i, algorithm in enumerate(['KMeans', 'KMeans++', 'XMeans']):
            avg_scores = [np.mean(scores[dataset][algorithm]) for dataset in dataset_names]
            bars = plt.bar(x_positions + i * bar_width, avg_scores, width=bar_width, color=colors[i], label=algorithm)

            # Annotate bars with exact values
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}', ha='center', fontsize=10)

        # Customize plot
        plt.title(f'{metric_name} by Clustering Algorithm and Dataset', fontsize=14)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(x_positions + bar_width, dataset_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    # Plot each metric
    plot_scores(silhouette_scores, "Silhouette Score")
    plot_scores(f1_scores, "F1-Score")
    plot_scores(jaccard_scores, "Jaccard Index")
    plot_scores(purity_scores, "Purity Index")
    plot_scores(rand_scores, "Rand Index")


def improved_kmeans_start(X, y, k, dataset_name):
    while True:
        print("\nChoose an Improved K-Means version:\n")
        print("\t1) KMeans++")
        print("\t2) X-KMeans")
        print("\t3) Comparison")
        print("\t4) Return to Main Menu")
        choice = input("\nYour choice: ")

        if choice == "1":
            kmeans_plus_plus(X, y, dataset_name=dataset_name)
        elif choice == "2":
            x_means(X, y, dataset_name=dataset_name)
        elif choice == "3":
            comparison_plot(k)
        elif choice == "4":
            print("Returning to the main menu.")
            break
        else:
            print("Invalid input. Please enter a valid option.")


def run_internal_analysis_improved(X, y, kmeans_type, dataset_name, max_iter=100, n_runs=5):
    results = {'euclidean': [], 'manhattan': [], 'cosine': []}
    k_values = list(range(2, 7))

    for distance_metric in tqdm(results.keys()):
        for k in k_values:
            silhouette_scores = []
            calinski_scores = []
            f1_scores = []
            rand_scores = []
            for _ in range(n_runs):
                if kmeans_type == "x_means":
                    kmeans = XMeans(k_min=min(k_values), k_max=max(k_values), distance_metric=distance_metric,
                                    max_iter=max_iter)
                elif kmeans_type == "KMeans++":
                    kmeans = KMeansPlusPlus(k=k, distance_metric=distance_metric, max_iter=max_iter)

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


def run_external_analysis_improved(X, y, dataset_name, kmeans_type):
    # Define the k values based on the dataset
    if dataset_name == "hepatitis":
        k1, k2, k3 = 2, 4, 6
    elif dataset_name == "mx":
        k1, k2, k3 = 2, 4, 6
    elif dataset_name == "breast":
        k1, k2, k3 = 2, 3, 5

    k_values = [k1, k2, k3]
    # Create DataFrames to store results
    results1 = pd.DataFrame(columns=['Distance', 'Purity', 'F1 score', 'Jaccard Index'],
                            index=[f'Model k={k}' for k in k_values])
    results2 = pd.DataFrame(columns=['Distance', 'Purity', 'F1 score', 'Jaccard Index'],
                            index=['Model k=(2,7)'])
    n_runs = 5
    # X-Means algorithm evaluation
    if kmeans_type == "x_means":
        all_labels = np.zeros((len(X), n_runs), dtype=int)
        for run_idx in range(n_runs):
            model1 = XMeans(k_min=2, k_max=7, distance_metric="cosine")
            model1.fit_predict(X)
            model1.compute_labels(X)
            all_labels[:, run_idx] = model1.labels

        final_labels = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels
        )

        # Update results table with the distance and k values
        results2.loc['Model k=(2,7)', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels, y)
        results2.loc['Model k=(2,7)', 'Distance'] = "Cosine"
        print(tabulate(results2, headers="keys", tablefmt="fancy_grid"))

    # KMeans++ algorithm evaluation
    elif kmeans_type == "KMeans++":
        all_labels_m1 = np.zeros((len(X), n_runs), dtype=int)
        all_labels_m2 = np.zeros((len(X), n_runs), dtype=int)
        all_labels_m3 = np.zeros((len(X), n_runs), dtype=int)
        for run_idx in range(n_runs):
            model1 = KMeansPlusPlus(k=k1, distance_metric="cosine")
            model1.fit_predict(X)
            model1.compute_labels(X)
            all_labels_m1[:, run_idx] = model1.labels

            model2 = KMeansPlusPlus(k=k2, distance_metric="cosine")
            model2.fit_predict(X)
            model2.compute_labels(X)
            all_labels_m2[:, run_idx] = model2.labels

            model3 = KMeansPlusPlus(k=k3, distance_metric="cosine")
            model3.fit_predict(X)
            model3.compute_labels(X)
            all_labels_m3[:, run_idx] = model3.labels

        final_labels_m1 = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m1
        )
        final_labels_m2 = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m2
        )
        final_labels_m3 = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=all_labels_m3
        )

        # Update results table with the distance and k values for each model
        results1.loc[f'Model k={k1}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m1, y)
        results1.loc[f'Model k={k1}', 'Distance'] = "Cosine"

        results1.loc[f'Model k={k2}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m2, y)
        results1.loc[f'Model k={k2}', 'Distance'] = "Cosine"

        results1.loc[f'Model k={k3}', 'Purity':'Jaccard Index'] = calculate_external_indices(final_labels_m3, y)
        results1.loc[f'Model k={k3}', 'Distance'] = "Cosine"

        print(tabulate(results1, headers="keys", tablefmt="fancy_grid"))
