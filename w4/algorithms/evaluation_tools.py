import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances, confusion_matrix, adjusted_rand_score, mutual_info_score, f1_score, \
    rand_score


def pairwise_dist_kde(X, dataset_name, save=False):
    # Estimate density of dataset
    kde = KernelDensity(bandwidth='scott')
    kde.fit(X)
    densities_estimates = kde.score_samples(X)
    actual_densities = np.exp(densities_estimates)
    avg_density = np.mean(actual_densities)
    # Plot pairwise distances distribution
    distances = pdist(X, metric='euclidean')
    sns.kdeplot(distances, color='green', fill=True, alpha=0.6)
    plt.text(0.6, 0.9, f'Dataset density: {avg_density:.3e}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
    plt.xlabel("Distance")
    plt.ylabel("Density")
    if save:
        plt.savefig(f'data/results/{dataset_name}_ds_density.png')
    plt.show()


def minkowski_mixed_distance(X, x, y, numerical_columns):  # Manhattan: r=1, Euclidean: r=2
    minkowski_distance = 0
    for i in range(len(x)):
        col = X.columns[i]
        if col in numerical_columns:
            minkowski_distance += abs(x[i] - y[i]) ** 2
        else:
            minkowski_distance += (1 - int(x[i] == y[i]))
    return minkowski_distance ** (1 / 2)


def gower_distance(X, x, y, numerical_columns):
    # Compute numerical feature ranges from the train set
    ranges = dict.fromkeys(numerical_columns)
    for num_col in numerical_columns:
        ranges[num_col] = max(X[num_col]) - min(X[num_col])
    # Compute Gower distance
    gower_distance = 0
    for i in range(len(x)):
        col = X.columns[i]
        if col in numerical_columns:
            gower_distance += abs(x[i] - y[i]) / ranges[col]
        else:
            gower_distance += (1 - int(x[i] == y[i]))
    return gower_distance / X.shape[1]


def calculate_dunn_score(X, labels, numerical_columns):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("At least two clusters are required to compute the Dunn index.")
    # Compute intra-cluster distances (diameter)
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = cdist(cluster_points, cluster_points,
                              metric=lambda x, y: gower_distance(X, x, y, numerical_columns))
            intra_cluster_distances.append(np.max(distances))
        else:
            intra_cluster_distances.append(0)  # Cluster with single point
    max_intra_distance = max(intra_cluster_distances)
    # Compute inter-cluster distances
    inter_cluster_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:  # Avoid duplicate pairs
                cluster_i_points = X[labels == label_i]
                cluster_j_points = X[labels == label_j]
                distances = cdist(cluster_i_points, cluster_j_points,
                                  metric=lambda x, y: gower_distance(X, x, y, numerical_columns))
                inter_cluster_distances.append(np.min(distances))

    min_inter_distance = min(inter_cluster_distances)

    return min_inter_distance / max_intra_distance


def calculate_jaccard_index(labels, true_labels):
    labels = np.array(labels)
    true_labels = np.array(true_labels)
    # Create pairwise binary comparison matrix
    true_pairs = np.equal.outer(true_labels, true_labels)
    pred_pairs = np.equal.outer(labels, labels)

    # Compute intersection and union
    intersection = np.logical_and(true_pairs, pred_pairs).sum()
    union = np.logical_or(true_pairs, pred_pairs).sum()

    # Return Jaccard Index
    return intersection / union


def calculate_purity(true_labels, labels):
    # Create confusion matrix (clusters x classes)
    contingency_matrix = confusion_matrix(true_labels, labels)

    # Find the maximum over rows (maximum class count for each cluster)
    max_in_cluster = np.amax(contingency_matrix, axis=0)

    # Sum the maximum counts and normalize by total number of samples
    purity = np.sum(max_in_cluster) / np.sum(contingency_matrix)

    return purity


def calculate_modularity(aff_matrix, labels, X=None, binarize=False):
    if binarize:
        aff_matrix = (aff_matrix > 0.8).astype(int)  # Create adjacency matrix based on threshold
    # Compute degree matrix
    aff_sum_rows = np.sum(aff_matrix, axis=1)
    degree_matrix = np.squeeze(np.asarray(aff_sum_rows))

    # Number of edges (total sum of the affinity matrix)
    m = np.sum(aff_matrix) / 2

    # Compute modularity
    modularity = 0
    n = aff_matrix.shape[0]  # number of nodes
    for i in range(n):
        for j in range(n):
            # Kronecker delta: 1 if nodes i and j are in the same community, 0 otherwise
            delta = 1 if labels[i] == labels[j] else 0
            # Modular contribution
            modularity += (aff_matrix[i, j] - (degree_matrix[i] * degree_matrix[j]) / (2 * m)) * delta

    # Normalize the modularity by dividing by 2m
    modularity = modularity / (2 * m)

    return modularity


def calculate_cluster_density_kde(X, labels):
    """
    Compute the density of clusters using Kernel Density Estimation (KDE).

    Parameters:
    X -> the dataset
    labels -> cluster labels for each data point.

    Returns:
    densities -> dictionary where keys are cluster labels, and values are average densities of the corresponding clusters.
    """
    clusters = list(set(labels))
    densities = {}

    for label in clusters:
        # Skip noise points
        if label == -1:
            continue
        # Extract points belonging to the cluster
        cluster_points = X[labels == label]
        # Fit KDE on the cluster points
        kde = KernelDensity(bandwidth='scott')
        kde.fit(cluster_points)
        # Estimate densities for points in the cluster
        densities_estimates = kde.score_samples(cluster_points)
        # Compute average density for the cluster
        avg_density = np.mean(np.exp(densities_estimates))  # Convert log-density to density
        densities[int(label)] = f'{avg_density:.3e}'

    return densities


def calculate_connectivity(X, labels, metric, numerical_columns):
    """
    Compute the connectivity of clusters based on pairwise distances.

    Parameters:
    - X -> the dataset
    - labels -> the cluster labels for each point
    - metric ->

    Returns:
    - connectivity -> connectivity score
    """
    dist_matrix = pairwise_distances(X, metric=lambda x, y: gower_distance(X, x, y, numerical_columns))
    clusters = np.unique(labels)
    # Exclude noise points
    clusters = clusters[clusters != -1]

    connectivity_score = 0
    for cluster in clusters:
        # Get indices of points in the current cluster
        cluster_indices = np.where(labels == cluster)[0]
        # Calculate the pairwise distances between points within the same cluster
        intra_cluster_distances = dist_matrix[cluster_indices][:, cluster_indices]
        # Count the number of points within the same cluster that are connected
        # by distances below a certain threshold (e.g., mean or median distance)
        threshold = np.median(intra_cluster_distances)
        # Iterate over the distances and count how many are below the threshold, excluding self connections
        connected_points = np.sum(intra_cluster_distances < threshold) - len(cluster_indices)
        connectivity_score += connected_points

    # Normalize by the total possible connections (excluding noise points)
    valid_points = np.sum(labels != -1)
    connectivity_score /= (valid_points * (valid_points - 1) / 2)

    return connectivity_score


def calculate_xie_beni_index(X, V, U, m):
    """
    Calculate the Xie-Beni Index

    Parameters:
    - X -> dataset for which we compute the Xie-Beni index
    - V -> cluster centers
    - U -> membership matrix
    - m -> fuzzifier parameter

    Returns:
    - XB: Xie-Beni Index
    """

    # compute numerator
    compactness = 0.0
    for i in range(V.shape[0]):
        for j in range(X.shape[0]):
            distance = np.linalg.norm(X.iloc[j].values - V[i]) ** 2
            compactness += (U[i, j] ** m) * distance

    # compute denominator
    min_separation = float('inf')
    for i in range(V.shape[0]):
        for k in range(i + 1, V.shape[0]):
            distance = np.linalg.norm(V[i] - V[k]) ** 2
            min_separation = min(min_separation, distance)

    XB = compactness / (X.shape[0] * min_separation)
    return XB


def compute_PE(U):
    """
    Compute the Partition Entropy (PE) of a fuzzy clustering

    Parameters:
    - U -> membership matrix
    """
    # PE
    pe = -np.sum(U * np.log(U + 1e-10)) / U.shape[0]
    return pe


def calculate_hamming_matrix(labels_dict):
    hamming_distance = lambda x, y: np.sum(x != y)
    hamming_matrix = np.zeros((len(labels_dict), len(labels_dict)))
    for i in range(len(labels_dict.keys())):
        for j in range(len(labels_dict.keys()))[i:]:
            if any([labels_dict[f"{i}"].size == 0, labels_dict[f"{j}"].size == 0]):
                continue
            hamming_matrix[i, j] = hamming_distance(labels_dict[f"{i}"], labels_dict[f"{j}"])

    return hamming_matrix


def plot_hamming_matrices(hamming_matrices, vmax, x_labels=None, k=None, dataset_name=None, save=False):
    if not isinstance(hamming_matrices, list):
        hamming_matrices = [hamming_matrices]
    fig_width = len(hamming_matrices) * 3
    fig_height = 3
    fig, ax = plt.subplots(1, len(hamming_matrices), figsize=(fig_width, fig_height))
    if len(hamming_matrices) == 1:
        ax = [ax]
    for i, hamming_matrix in enumerate(hamming_matrices):
        # Create a heatmap
        mask = np.tril(np.ones_like(hamming_matrix, dtype=bool))
        sns.heatmap(hamming_matrix, mask=mask, cmap="Reds", square=True, linewidths=0.5, cbar=True, vmin=0, vmax=vmax,
                    ax=ax[i])
        if x_labels:
            ax[i].set_xticks(np.arange(hamming_matrix.shape[1]) + 0.5)
            ax[i].set_xticklabels(x_labels, rotation=45, ha="left", fontsize=8)
            ax[i].xaxis.set_ticks_position("top")
        else:
            ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    if save:
        plt.savefig(f'data/results/{dataset_name}_hamming.png')
    plt.show()


def plot_reachability_distances(reachability_distances, ordering, labels_pred, dataset_name, save=False):
    clusters = list(set(labels_pred))

    plt.figure(figsize=(10, 5))
    colormap = cm.get_cmap('tab10', len(clusters))
    for i, lbl in enumerate(sorted(clusters)):
        if lbl == -1:
            label = 'noise'
            color = 'black'
        else:
            label = f'cluster {i}'
            color = colormap(i)
        cluster_pts = labels_pred == lbl
        cluster_reachability_dist = reachability_distances[ordering[cluster_pts]]
        plt.scatter(range(np.sum(cluster_pts)), cluster_reachability_dist, color=color, s=10, label=label)
    plt.xlabel('Ordering of points')
    plt.ylabel('Reachability Distance')
    plt.xticks([])
    plt.title('Reachability Plot')
    plt.legend()
    if save:
        plt.savefig(f'data/results/{dataset_name}_reachability.png')
    plt.show()


def plot_metrics(results, n_metrics, metric_titles, k_values, colors, colors_labels_map, dataset_name, save=False):
    """
    Plots different metrics for different values of k and for different parameter configurations.

    Parameters:
    - results -> dictionary that contains combination of parameters as keys, and a matrices of dimension
    n_metrics x k_values , in which each row is a vector that contains metric values for each k value
    - n_metrics -> Number of metrics that will be plotted, corresponds to number of subplots
    - metric_titles -> titles for the metrics. Its length = n_metrics
    - k_values -> x-axis of each subplot
    - colors -> dictionary that contains combination of parameters as key, corresponding to a line of the subplots,
    and color for that combination/line as value
    - colors_labels_map -> dictionary to assign a label to each color since we might want to plot several parameter
    combinations with the same color
    - dataset_name -> name of the dataset for which plot are being computed. Used if save = True
    - save -> saves plots in the data/results folder

    Returns:
    - None
    """

    fig_width = n_metrics * 5
    fig_height = 5
    fig, axs = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height))

    # If there is only one metric, axs will not be an array, so we wrap it in a list
    if n_metrics == 1:
        axs = [axs]

    # Create an empty list to store lines and labels for the legend
    lines = []
    labels = []

    for m in range(n_metrics):
        for n_comb, metrics in results.items():
            metric_values = metrics[m]
            label = f'{colors_labels_map[colors[n_comb]]}'
            # Plot the line with the label
            line, = axs[m].plot(k_values, metric_values, marker='o', color=colors[n_comb], label=label)
            # Add this line to the list only if the label is not already added
            if label not in labels:
                lines.append(line)
                labels.append(label)
        axs[m].set_xlabel('Number of Clusters (k)')
        axs[m].set_xticks(k_values)
        axs[m].set_ylabel(metric_titles[m])
        # axs[m].set_aspect('equal')
        axs[m].grid(True)

    axs[0].legend(lines, labels, loc='best', ncol=1, fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(f'data/results/{dataset_name}_metrics.png')
    plt.show()


def plot_clusters(X, dataset_name, clusters, centroids=[], save=False):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for cluster, indices in clusters.items():
        cluster_points = X_pca[indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    if len(centroids) > 0:
        centroids_pca = pca.transform(centroids)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='red', label='Centroids')
        title = "Clusters and Centroids"
    else:
        title = "Clusters"
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f'data/results/{dataset_name}_clusters_pca.png')
    plt.show()


def calculate_external_indices(labels_pred, labels_true):
    # Purity
    purity = calculate_purity(labels_true, labels_pred)
    # Adjusted rand index
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Rand index
    ri = rand_score(labels_true, labels_pred)
    # Mutual information
    mi = mutual_info_score(labels_true, labels_pred)
    # Jaccard index
    jacc = calculate_jaccard_index(labels_pred, labels_true)
    # F1 score
    f1score = f1_score(labels_true, labels_pred, average='weighted')

    return [purity, ari, ri, jacc]


def plot_confusion_matrix_heatmap(labels_true, labels_pred, dataset_name, save=False):
    cm = confusion_matrix(labels_true, labels_pred)
    class_names = sorted(set(labels_true) | set(labels_pred))
    class_names = [int(c) for c in class_names]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    if save:
        plt.savefig(f'data/results/{dataset_name}_confusion_mat.png')
    plt.show()
