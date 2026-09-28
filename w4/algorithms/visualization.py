import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.pca import pca_n_components, pca_variance


def visualize_clustering_pca(X, y, features, dataset_name):

    # Clustering with K-Means++
    kmeans = KMeansPlusPlus(k=2, distance_metric='manhattan')
    kmeans.fit_predict(X.to_numpy())

    # Clustering with Optics
    optics = OPTICS(min_samples=15, metric='l1', xi=0.05, min_cluster_size=0.05)
    optics_labels = optics.fit_predict(X)

    # Reduce dimensionality to visualize with PCA
    _, _, _, _, X_pca = pca_n_components(X, features, view_plots=False, n_components=2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # PCA to visualize without prior PCA to reduce
    # 1. Dataset original w/ PCA
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c="gray", s=20)
    axes[0, 0].set_title("Original Dataset (PCA Visualization)")
    axes[0, 0].set_xlabel("Component 1")
    axes[0, 0].set_ylabel("Component 2")

    # Shared Color Map for Both KMeans++ and OPTICS
    unique_kmeans_clusters = np.unique(list(kmeans.clusters.keys()))
    unique_optics_clusters = np.unique(optics_labels)

    # Combine unique cluster labels from KMeans and OPTICS
    unique_clusters = np.union1d(unique_kmeans_clusters, unique_optics_clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Create a mapping of cluster labels to colors
    cluster_colors = {
        cluster: color
        for cluster, color in zip(unique_clusters, colors)
    }

    # 2. KMeans++ Clustering
    axes[0, 1].set_title("KMeans++ Clustering (PCA Visualization)")
    axes[0, 1].set_xlabel("Component 1")
    axes[0, 1].set_ylabel("Component 2")

    for cluster, indices in kmeans.clusters.items():
        cluster_points = X_pca[indices]
        color = cluster_colors[cluster]
        axes[0, 1].scatter(cluster_points[:, 0], cluster_points[:, 1], c=color.reshape(1, -1),
                           label=f'Cluster {cluster}', s=20)

    # 3. OPTICS Clustering
    axes[0, 2].set_title("OPTICS Clustering (PCA Visualization)")
    axes[0, 2].set_xlabel("Component 1")
    axes[0, 2].set_ylabel("Component 2")

    for cluster in unique_optics_clusters:
        cluster_points = X_pca[optics_labels == cluster]
        if cluster == -1:  # Noise points
            axes[0, 2].scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c="gray",
                label="Noise",
                s=20,
                alpha=0.6
            )
        else:
            color = cluster_colors[cluster]
            axes[0, 2].scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=color.reshape(1, -1),
                label=f"Cluster {cluster}",
                s=20,
                alpha=0.8
            )

    # Add legends to both subplots
    axes[0, 1].legend(loc="best", title="KMeans++ Clusters")
    axes[0, 2].legend(loc="best", title="OPTICS Clusters")

    # Row 2 Reduction with PCA and Visualization with PCA
    _, _, _, X_pca_reduced, _ = pca_variance(X, features, view_plots=False, captured_variance=0.8)
    # Clustering with K-Means++
    kmeans_reduced = KMeansPlusPlus(k=2, distance_metric='manhattan')
    kmeans_reduced.fit_predict(X_pca_reduced)

    # Clustering with Optics
    optics = OPTICS(min_samples=15, metric='l1', xi=0.05, min_cluster_size=0.05)
    pca_optics_labels = optics.fit_predict(X_pca_reduced)

    df = pd.DataFrame(X_pca_reduced, columns=X.columns)
    _, _, _, _, X_pca2 = pca_n_components(df, features, view_plots=False, n_components=2)

    # 4. Dataset reduced con PCA
    axes[1, 0].scatter(X_pca2[:, 0], X_pca2[:, 1], c="gray", s=20)
    axes[1, 0].set_title("Original Dataset (PCA Reduced)")
    axes[1, 0].set_xlabel("Component 1")
    axes[1, 0].set_ylabel("Component 2")

    # Shared Color Map for Both KMeans++ and OPTICS (Reduced PCA)
    unique_kmeans_clusters_reduced = np.unique(list(kmeans_reduced.clusters.keys()))
    unique_optics_clusters_reduced = np.unique(pca_optics_labels)

    # Combine unique cluster labels from KMeans and OPTICS (reduced PCA)
    unique_clusters_reduced = np.union1d(unique_kmeans_clusters_reduced, unique_optics_clusters_reduced)
    colors_reduced = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters_reduced)))

    # Create a mapping of cluster labels to colors
    cluster_colors_reduced = {
        cluster: color
        for cluster, color in zip(unique_clusters_reduced, colors_reduced)
    }

    # 5. KMeans++ Clustering with PCA
    axes[1, 1].set_title("KMeans++ Clustering (PCA Reduced)")
    axes[1, 1].set_xlabel("Component 1")
    axes[1, 1].set_ylabel("Component 2")

    for cluster, indices in kmeans_reduced.clusters.items():
        cluster_points = X_pca2[indices]
        color = cluster_colors_reduced[cluster]
        axes[1, 1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=color.reshape(1, -1),
            label=f"Cluster {cluster}",
            s=20
        )

    # 6. OPTICS Clustering with PCA
    axes[1, 2].set_title("OPTICS Clustering (PCA Reduced)")
    axes[1, 2].set_xlabel("Component 1")
    axes[1, 2].set_ylabel("Component 2")

    for cluster in unique_optics_clusters_reduced:
        cluster_points = X_pca2[pca_optics_labels == cluster]
        if cluster == -1:  # Noise points
            axes[1, 2].scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c="gray",
                label="Noise",
                s=20,
                alpha=0.6
            )
        else:
            color = cluster_colors_reduced[cluster]
            axes[1, 2].scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=color.reshape(1, -1),
                label=f"Cluster {cluster}",
                s=20,
                alpha=0.8
            )

    # Add legends for both subplots
    axes[1, 1].legend(loc="best", title="KMeans++ Clusters")
    axes[1, 2].legend(loc="best", title="OPTICS Clusters")

    axes[1, 0].set_xlim(axes[1, 0].get_xlim()[::-1])
    axes[1, 1].set_xlim(axes[1, 1].get_xlim()[::-1])
    axes[1, 2].set_xlim(axes[1, 2].get_xlim()[::-1])

    plt.suptitle(f"Clustering Visualization for {dataset_name}")
    plt.tight_layout()
    plt.show()


def visualize_clustering_umap(X, y, features, dataset_name):
    # Clustering with K-Means++
    kmeans = KMeansPlusPlus(k=2, distance_metric='manhattan')
    kmeans.fit_predict(X.to_numpy())
    kmeans.compute_labels(X.to_numpy())
    # Clustering with Optics
    optics = OPTICS(min_samples=15, metric='l1', xi=0.05, min_cluster_size=0.05)
    optics_labels = optics.fit_predict(X)

    # Reduce dimensionality using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=35)
    X_umap = umap_model.fit_transform(X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Dataset original con UMAP
    axes[0, 0].scatter(X_umap[:, 0], X_umap[:, 1], c="gray", s=20)
    axes[0, 0].set_title("Original Dataset (UMAP Visualization)")
    axes[0, 0].set_xlabel("UMAP1")
    axes[0, 0].set_ylabel("UMAP2")

    # 2. KMeans++ Clustering with UMAP
    scatter = axes[0, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=kmeans.labels, cmap="tab10", s=20)
    axes[0, 1].set_title("KMeans++ Clustering (UMAP Visualization)")
    axes[0, 1].set_xlabel("UMAP1")
    axes[0, 1].set_ylabel("UMAP2")

    # Custom legend labels for KMeans++
    labels_kmeans = ['Noise points' if label == -1 else f'{int(label)} cluster' for label in np.unique(kmeans.labels)]
    handles_kmeans, _ = scatter.legend_elements()
    axes[0, 1].legend(handles_kmeans, labels_kmeans, title="Clusters")

    # 3. OPTICS Clustering
    scatter = axes[0, 2].scatter(X_umap[:, 0], X_umap[:, 1], c=optics_labels, cmap="tab10", s=20)
    axes[0, 2].set_title("OPTICS Clustering (UMAP Visualization)")
    axes[0, 2].set_xlabel("UMAP1")
    axes[0, 2].set_ylabel("UMAP2")

    # Custom legend labels for OPTICS
    labels_optics = ['Noise points' if label == -1 else f'{label} cluster' for label in np.unique(optics_labels)]
    handles_optics, _ = scatter.legend_elements()
    axes[0, 2].legend(handles_optics, labels_optics, title="Clusters")

    _, _, _, X_pca_reduced, _ = pca_variance(X, features, view_plots=False, captured_variance=0.9)

    # Clustering with K-Means++
    kmeans_reduced = KMeansPlusPlus(k=2, distance_metric='manhattan')
    kmeans_reduced.fit_predict(X_pca_reduced)
    kmeans_reduced.compute_labels(X_pca_reduced)

    # Clustering with Optics
    optics_reduced = OPTICS(min_samples=15, metric='l1', xi=0.05, min_cluster_size=0.05)
    optics_reduced_labels = optics_reduced.fit_predict(X_pca_reduced)

    # Apply UMAP to reduced data
    umap_model_reduced = umap.UMAP(n_components=2, random_state=35)
    X_umap_reduced = umap_model_reduced.fit_transform(X_pca_reduced)

    # 4. Dataset reduced w/ UMAP
    axes[1, 0].scatter(X_umap_reduced[:, 0], X_umap_reduced[:, 1], c="gray", s=20)
    axes[1, 0].set_title("Dataset (UMAP Reduced)")
    axes[1, 0].set_xlabel("UMAP1")
    axes[1, 0].set_ylabel("UMAP2")

    # 5. KMeans++ Clustering reduced w/UMAP
    scatter = axes[1, 1].scatter(X_umap_reduced[:, 0], X_umap_reduced[:, 1], c=kmeans_reduced.labels, cmap="tab10",
                                 s=20)
    axes[1, 1].set_title("KMeans++ Clustering (UMAP Reduced)")
    axes[1, 1].set_xlabel("UMAP1")
    axes[1, 1].set_ylabel("UMAP2")

    # Legend labels for KMeans++ (reduced)
    labels_kmeans_reduced = ['Noise points' if label == -1 else f'{int(label)} cluster' for label in
                             np.unique(kmeans_reduced.labels)]
    handles_kmeans_reduced, _ = scatter.legend_elements()
    axes[1, 1].legend(handles_kmeans_reduced, labels_kmeans_reduced, title="Clusters")

    # 6. OPTICS Clustering
    scatter = axes[1, 2].scatter(X_umap_reduced[:, 0], X_umap_reduced[:, 1], c=optics_reduced_labels, cmap="tab10",
                                 s=20)
    axes[1, 2].set_title("OPTICS Clustering (UMAP Reduced)")
    axes[1, 2].set_xlabel("UMAP1")
    axes[1, 2].set_ylabel("UMAP2")

    # Legend labels for OPTICS (reduced)
    labels_optics_reduced = ['Noise points' if label == -1 else f'{label} cluster' for label in
                             np.unique(optics_reduced_labels)]
    handles_optics_reduced, _ = scatter.legend_elements()
    axes[1, 2].legend(handles_optics_reduced, labels_optics_reduced, title="Clusters")

    plt.suptitle(f"Clustering Visualization for {dataset_name}")
    plt.tight_layout()
    plt.show()


def input_visualization(X, y, features, dataset_name):
    while True:
        visualization_method = input("Type the visualization method ('pca' or 'umap') or 'Exit' to go back to the menu: ")
        if visualization_method not in ["pca", "umap", "Exit"]:
            print("Invalid input format. Please try again.")
        elif visualization_method == "pca":
            visualize_clustering_pca(X, y, features, dataset_name)
        elif visualization_method == "umap":
            visualize_clustering_umap(X, y, features, dataset_name)
        elif visualization_method == "Exit":
            break