import numpy as np

from algorithms.pca_plots import pca_plot_original_space, transformed_space_plot


def pca_variance(df, features, view_plots=True, captured_variance=0.9):
    # Store the indices of the features
    selected_feature_indices = [df.columns.get_loc(f) for f in features]

    # Compute the mean vector
    mean_vector = np.mean(df.values, axis=0)

    # Compute the covariance matrix
    data_centered = df.values - mean_vector
    covariance_matrix = np.cov(data_centered, rowvar=False)

    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the total variance
    total_variance = np.sum(sorted_eigenvalues)

    # Calculate the cumulative variance explained
    cumulative_variance = np.cumsum(sorted_eigenvalues) / total_variance

    # Find the number of components to capture 90% of the variance
    k = np.argmax(cumulative_variance >= captured_variance) + 1

    # Select the top-k eigenvectors (one column per vector9
    eigenvectors_k = sorted_eigenvectors[:, :k]

    # Derive the new data set
    transformed_data = np.dot(data_centered, eigenvectors_k)

    # Reconstruct the data set back to the original one
    reconstructed_data = np.dot(transformed_data, eigenvectors_k.T) + mean_vector

    # Select the columns from the reconstructed data that correspond to the original selected features
    selected_reconstructed_data = reconstructed_data[:, selected_feature_indices]

    # Explained variance ratio
    explained_variance_ratio = sorted_eigenvalues[:k] / total_variance

    # Reconstruction error
    reconstruction_error = np.mean((df[features].values - selected_reconstructed_data) ** 2)

    if view_plots:
        print("\nCovariance Matrix:")
        print(covariance_matrix)

        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors:")
        print(eigenvectors)

        print("\nSorted Eigenvalues:")
        print(sorted_eigenvalues)
        print("\nSorted Eigenvectors:")
        print(sorted_eigenvectors)

        transformed_space_plot(transformed_data, k, variance=captured_variance)

        pca_plot_original_space(selected_reconstructed_data, features,
                                f"Reconstructed Dataset Features with {captured_variance}% variance")
    return k, reconstruction_error, explained_variance_ratio, reconstructed_data, transformed_data


def pca_n_components(df, features, view_plots=True, n_components=2):
    # Store the indices of the features
    selected_feature_indices = [df.columns.get_loc(f) for f in features]

    # Compute the mean vector
    mean_vector = np.mean(df.values, axis=0)

    # Compute the covariance matrix
    data_centered = df.values - mean_vector
    covariance_matrix = np.cov(data_centered, rowvar=False)

    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the total variance
    total_variance = np.sum(sorted_eigenvalues)

    # Select the top-n_components eigenvectors (one column per vector)
    eigenvectors_k = sorted_eigenvectors[:, :n_components]

    # Derive the new data set
    transformed_data = np.dot(data_centered, eigenvectors_k)

    # Reconstruct the data set back to the original one
    reconstructed_data = np.dot(transformed_data, eigenvectors_k.T) + mean_vector

    # Select the columns from the reconstructed data that correspond to the original selected features
    selected_reconstructed_data = reconstructed_data[:, selected_feature_indices]

    # Explained variance ratio
    explained_variance_ratio = sorted_eigenvalues[:n_components] / total_variance

    # Reconstruction error
    reconstruction_error = np.mean((df[features].values - selected_reconstructed_data) ** 2)

    if view_plots:
        print("\nMean Vector:")
        print(mean_vector)

        print("\nCovariance Matrix:")
        print(covariance_matrix)

        print("\nSorted Eigenvalues:")
        print(sorted_eigenvalues)
        print("\nSorted Eigenvectors:")
        print(sorted_eigenvectors)

        print(f"\nNumber of components selected: {n_components}")

        print("\nTransformed Data:")
        print(transformed_data)

        # Plot the new subspace (use the first two principal components for 2D visualization)
        transformed_space_plot(transformed_data, n_components)

        print("\nReconstructed Data:")
        print(reconstructed_data)

        pca_plot_original_space(
            selected_reconstructed_data,
            features,
            f"Reconstructed Dataset Features with {n_components} components",
        )
    return n_components, reconstruction_error, explained_variance_ratio, reconstructed_data, transformed_data
