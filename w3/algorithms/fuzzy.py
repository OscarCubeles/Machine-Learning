import itertools

import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import cm
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from algorithms.evaluation_tools import plot_metrics, calculate_external_indices, calculate_xie_beni_index, \
    compute_PE


class Fuzzy:
    def __init__(self, X, dataset_name, k_values, fuzzifier_parameter, suppression_parameter):
        self.X = X
        self.dataset_name  =dataset_name
        self.k_values = k_values
        self.fuzzifier_parameter = fuzzifier_parameter
        self.suppression_parameter = suppression_parameter
        self.threshold = 1e-3
        self.max_iter = 200
        self.clusters = None

    def fit_predict(self):
        stopiter = 1
        n_loops = 0
        U = self.initialize_membership_matrix()
        V = self.initialize_centroids()
        while stopiter > self.threshold and n_loops <= self.max_iter:  # Iterative process.
            U = self.update_membership(V)
            V_new = self.compute_centroids(U)
            stopiter = np.linalg.norm(V - V_new)
            V = V_new
            n_loops += 1

        # compute labels = max columns U
        labels_pred = np.argmax(U, axis=0)

        # Get clusters
        clusters_dict = {}
        for cluster_label in np.unique(labels_pred):
            # Skip noise points
            if cluster_label != -1:
                cluster_indices = np.where(labels_pred == cluster_label)[0]
                clusters_dict[cluster_label] = cluster_indices
        self.clusters = clusters_dict

        return U, V, labels_pred

    def initialize_membership_matrix(self):
        # Randomly initialize membership matrix U
        U = np.random.rand(self.k_values, self.X.shape[0])
        U = U / np.sum(U, axis=0)
        return U

    def initialize_centroids(self):
        # initialize centroids with random columns of X
        U = self.X.to_numpy()
        rows = np.random.choice(self.X.shape[0], self.k_values, replace=False)
        U = U[rows, :]
        return U

    def compute_centroids(self, U):
        # Compute centroids
        V = []
        for i in range(self.k_values):
            matrix = np.power(U[i, :], self.fuzzifier_parameter)
            denominator = np.sum(np.where(np.isfinite(matrix), matrix, 0))
            numerator = (U[i, :] ** self.fuzzifier_parameter).reshape(-1, 1) * self.X
            V.append(np.sum(numerator, axis=0) / denominator)
        return np.array(V)

    def update_membership_v1(self, V):
        # Update the membership matrix U
        U = np.zeros((self.k_values, self.X.shape[0]))
        for i in range(self.k_values):
            for k in range(self.X.shape[0]):
                x_k = self.X.iloc[k].values
                v_i = V[i]
                # numerator ||x_k - v_i|| using euclidean distance
                numerator = np.linalg.norm(x_k - v_i) ** 2
                aux = np.zeros((self.k_values,))
                for j in range(self.k_values):
                    denominator = np.linalg.norm(x_k - V[j]) ** 2
                    aux[j] = numerator / denominator + 1e-10

                res = np.sum(np.power(aux, 2 / (self.fuzzifier_parameter - 1)))

                U[i][k] = 1 / res
        return U

    def update_membership(self, V):
        # Update the membership matrix U
        U = np.zeros((self.k_values, self.X.shape[0]))
        for i in range(self.k_values):
            for k in range(self.X.shape[0]):
                x_k = self.X.iloc[k].values

                distances = np.array([np.linalg.norm(x_k - v) ** 2 for v in V])
                # Compute suppression term
                a_k = self.suppression_parameter * np.min(distances)

                numerator = distances[i] - a_k
                aux = np.zeros((self.k_values,))
                for j in range(self.k_values):
                    denominator = distances[j] - a_k
                    aux[j] = numerator / denominator + 1e-10

                res = np.sum(np.power(aux, 2 / (self.fuzzifier_parameter - 1)))

                U[i][k] = 1 / res
        return U

def run_internal_analysis(X, dataset_name, k_values):
    fuzzifier_parameters = [1.01, 1.3, 1.6, 1.9, 2.2]
    suppression_parameters = [0.99]
    results = {}
    comb_param = {}
    n_comb = 0
    for m in fuzzifier_parameters:
        for w in suppression_parameters:
            comb_param[n_comb] = [m, w]
            for k in tqdm(k_values):
                fuzzy = Fuzzy(X, dataset_name, k, m, w)
                U, V, labels_pred = fuzzy.fit_predict()

                # Compute Silhouette, Xie-Beni Index and geometric mean between Partition Coefficient and Partition Entropy
                sil = silhouette_score(X, labels_pred)
                xb = calculate_xie_beni_index(X, V, U, m)
                pe = compute_PE(U)
                metrics = np.array([sil, xb, pe])

                # Save metrics
                if k == k_values[0]:
                    results[n_comb] = metrics.reshape(-1, 1)
                else:
                    results[n_comb] = np.hstack([results[n_comb], metrics.reshape(-1, 1)])

            n_comb += 1

    # Plot metrics
    combinations = list(itertools.product(suppression_parameters, fuzzifier_parameters))
    colormap = cm.get_cmap('tab20', len(combinations))
    colors = {}

    if len(suppression_parameters) > 1:
        colors_labels_map = {colormap(i) : f"w={suppression}" for i, suppression in enumerate(suppression_parameters)}
        for n_comb in comb_param.keys():
            sup = comb_param[n_comb][1]
            colors[n_comb] = next((k for k, v in colors_labels_map.items() if v ==  f"w={sup}"), None)
    else:
        colors_labels_map = {colormap(i): f"w={suppression_parameters[0]}, m={param}" for i, param in enumerate(fuzzifier_parameters)}
        for i, n_comb in enumerate(comb_param.keys()):
            colors[n_comb] = list(colors_labels_map.keys())[i]

    metric_titles = ['Silhouette', 'Xie-Beni', 'Partition Entropy']
    plot_metrics(results, len(metrics), metric_titles, k_values, colors, colors_labels_map, dataset_name)

def run_external_analysis(X, y, dataset_name):
    # Define values based on the dataset
    if dataset_name == "hepatitis":
        k1, k2, k3 = 2, 2, 6
        w = 0.99
        m1, m2, m3 = 1.01, 1.3, 1.01
    elif dataset_name == "mx":
        k1, k2, k3 = 2, 4, 6
        w = 0.99
        m1, m2, m3 = 1.01, 2.2, 1.01
    elif dataset_name == "breast":
        k1, k2, k3 = 2, 2, 5
        w = 0.99
        m1, m2, m3 = 1.01, 1.3, 1.01

    # Initialize and fit the KMeans models
    fuzzy1 = Fuzzy(X, dataset_name, k1, m1, w)
    _, _, labels_pred1 = fuzzy1.fit_predict()

    fuzzy2 = Fuzzy(X, dataset_name, k2, m2, w)
    _, _, labels_pred2 = fuzzy2.fit_predict()

    fuzzy3 = Fuzzy(X, dataset_name, k3, m3, w)
    _, _, labels_pred3 = fuzzy3.fit_predict()

    # Create DataFrame to store results with additional columns for 'Distance' and 'k'
    results = pd.DataFrame(columns=['Purity', 'Adjusted Rand Index', 'Jaccard Index', 'k', 'm'],
                           index=[f'Model k={k1}, m={m1}', f'Model k={k2}, m={m2}', f'Model k={k3}, m={m3}'])

    # Calculate external indices and populate the table with results, distance, and k values

    results.loc[f'Model k={k1}, m={m1}', 'Purity':'Jaccard Index'] = calculate_external_indices(labels_pred1, y)
    results.loc[f'Model k={k1}, m={m1}', 'k'] = k1
    results.loc[f'Model k={k1}, m={m1}', 'm'] = m1

    results.loc[f'Model k={k2}, m={m2}', 'Purity':'Jaccard Index'] = calculate_external_indices(labels_pred2, y)
    results.loc[f'Model k={k2}, m={m2}', 'k'] = k2
    results.loc[f'Model k={k2}, m={m2}', 'm'] = m2

    results.loc[f'Model k={k3}, m={m3}', 'Purity':'Jaccard Index'] = calculate_external_indices(labels_pred3, y)
    results.loc[f'Model k={k3}, m={m3}', 'k'] = k3
    results.loc[f'Model k={k3}, m={m3}', 'm'] = m3

    # Print the results in a fancy grid table format
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

def get_fuzzy_input():
    while True:
        print("\nPlease enter your command in the following format:")
        print("<k>-<fuzzifier>-<suppression> (e.g., 3-1.2-0.9)")
        user_input = input("Enter your command: ").strip()
        try:
            k_str, fuzzifier_str, suppression_str = user_input.split('-')
            k = int(k_str)
            f = float(fuzzifier_str)
            s = float(suppression_str)
            if f < 1:
                print(f"Error: Invalid fuzzifier value. Choose fuzzifier > 1. Recommmended from (1, 2.5].")
                continue

            if s not in [0.9, 0.99]:
                print(f"Error: Invalid suppression value. Choose a value from [0.9, 0.99].")
                continue

            return k, f, s
        except ValueError:
            print("Error: Invalid input format. Please use the format <k>-<fuzzifier>-<suppression>.")
