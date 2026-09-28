import time
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA, IncrementalPCA
from algorithms.pca import pca_variance, pca_n_components
from algorithms.pca_plots import pca_plot_original_space, comparison_pca_plot, plot_pca_metrics


def pca_analysis(df, features):
    n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # PCA Plot 3 features from original space dataset
    pca_plot_original_space(df[features].values, features, "Original Dataset Features")
    pca_variance(df, features, captured_variance=0.95, view_plots=True)

    # Arrays to store metrics
    reconstruction_errors = []
    explained_variance_ratios = []

    # Analyze PCA for different variances
    for components in n_components:
        k, reconstruction_error, explained_variance_ratio, _, _ = pca_n_components(
            df, features, n_components=components, view_plots=False
        )
        reconstruction_errors.append(reconstruction_error)
        explained_variance_ratios.append(
            np.sum(explained_variance_ratio))

    # Bar plots for metrics
    plot_pca_metrics(n_components, reconstruction_errors, explained_variance_ratios)


def pca_comparison(df, features):
    n_components_list = [2, 3, 4, 5]
    methods = ["PCA", "Sklearn PCA", "Sklearn IPCA"]
    reconstruction_errors = {method: [] for method in methods}
    explained_variance_ratios = {method: [] for method in methods}
    times = {method: [] for method in methods}

    # Custom PCA
    for n_components in n_components_list:
        start_time = time.time()
        _, reconstruction_error, explained_variance_ratio, _, _ = pca_n_components(
            df, features, n_components=n_components, view_plots=False
        )
        end_time = time.time()

        reconstruction_errors["PCA"].append(reconstruction_error)
        explained_variance_ratios["PCA"].append(np.sum(explained_variance_ratio))
        times["PCA"].append(end_time - start_time)

    # Sklearn PCA
    for n_components in n_components_list:
        start_time = time.time()
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(df.values)
        reconstructed_data = pca.inverse_transform(transformed_data)
        end_time = time.time()

        reconstruction_error = np.mean((df.values - reconstructed_data) ** 2)
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

        reconstruction_errors["Sklearn PCA"].append(reconstruction_error)
        explained_variance_ratios["Sklearn PCA"].append(explained_variance_ratio)
        times["Sklearn PCA"].append(end_time - start_time)

    # Sklearn Incremental PCA
    for n_components in n_components_list:
        start_time = time.time()
        ipca = IncrementalPCA(n_components=n_components)
        transformed_data = ipca.fit_transform(df.values)
        reconstructed_data = ipca.inverse_transform(transformed_data)
        end_time = time.time()

        reconstruction_error = np.mean((df.values - reconstructed_data) ** 2)
        explained_variance_ratio = np.sum(ipca.explained_variance_ratio_)

        reconstruction_errors["Sklearn IPCA"].append(reconstruction_error)
        explained_variance_ratios["Sklearn IPCA"].append(explained_variance_ratio)
        times["Sklearn IPCA"].append(end_time - start_time)

    comparison_pca_plot(n_components_list, reconstruction_errors, explained_variance_ratios, times)

    return reconstruction_errors, explained_variance_ratios, times


def plot_comparison_pca_3d(X, features, n_components):
    # Store the indices of the features
    selected_feature_indices = [X.columns.get_loc(f) for f in features]

    fig = plt.figure(figsize=(20, 5))

    # 3D Plot for Our PCA
    _, _, _, reconstructed_data, _ = pca_n_components(
        X, features, n_components=n_components, view_plots=False
    )
    # Select the columns from the reconstructed data that correspond to the original selected features
    selected_reconstructed_data = reconstructed_data[:, selected_feature_indices]
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(selected_reconstructed_data[:, 0], selected_reconstructed_data[:, 1], selected_reconstructed_data[:, 2], alpha=0.7, edgecolor='k')
    ax1.set_title("Our PCA")
    ax1.set_xlabel(features[0])
    ax1.set_ylabel(features[1])
    ax1.set_zlabel(features[2])

    # 3D Plot for sklearn's PCA
    pca = PCA(n_components=n_components)
    transformed_data2 = pca.fit_transform(X.values)
    reconstructed_data2 = pca.inverse_transform(transformed_data2)
    selected_reconstructed_data2 = reconstructed_data2[:, selected_feature_indices]
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(selected_reconstructed_data2[:, 0], selected_reconstructed_data2[:, 1], selected_reconstructed_data2[:, 2], alpha=0.7, edgecolor='k')
    ax2.set_title("PCA")
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
    ax2.set_zlabel(features[2])

    # 3D Plot for sklearn's IPCA
    ipca = IncrementalPCA(n_components=n_components)
    transformed_data = ipca.fit_transform(X.values)
    reconstructed_data = ipca.inverse_transform(transformed_data)
    selected_reconstructed_data = reconstructed_data[:, selected_feature_indices]
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(selected_reconstructed_data[:, 0], selected_reconstructed_data[:, 1], selected_reconstructed_data[:, 2], alpha=0.7, edgecolor='k')
    ax3.set_title("IPCA")
    ax3.set_xlabel(features[0])
    ax3.set_ylabel(features[1])
    ax3.set_zlabel(features[2])

    fig.suptitle(f"3D Plot ({features[0]}, {features[1]}, {features[2]})", fontsize=16)
    plt.tight_layout()
    plt.show()
