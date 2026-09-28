import numpy as np
from matplotlib import pyplot as plt


def pca_plot_original_space(data_matrix, features, title=""):
    fig = plt.figure(figsize=(20, 5))

    # 3D Plot
    ax1 = fig.add_subplot(1, 4, 4, projection='3d')
    ax1.scatter(data_matrix[:, 0], data_matrix[:, 1], data_matrix[:, 2], alpha=0.7, edgecolor='k')
    ax1.set_title(f"3D Plot ({features[0]}, {features[1]}, {features[2]})")
    ax1.set_xlabel(features[0])
    ax1.set_ylabel(features[1])
    ax1.set_zlabel(features[2])

    # 2D Plot: Feature 1 vs Feature 2
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(data_matrix[:, 0], data_matrix[:, 1], alpha=0.7, edgecolor='k')
    ax2.set_title(f"{features[0]} vs {features[1]}")
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])

    # 2D Plot: Feature 1 vs Feature 3
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(data_matrix[:, 0], data_matrix[:, 2], alpha=0.7, edgecolor='k')
    ax3.set_title(f"{features[0]} vs {features[2]}")
    ax3.set_xlabel(features[0])
    ax3.set_ylabel(features[2])

    # 2D Plot: Feature 2 vs Feature 3
    ax4 = fig.add_subplot(1, 4, 1)
    ax4.scatter(data_matrix[:, 1], data_matrix[:, 2], alpha=0.7, edgecolor='k')
    ax4.set_title(f"{features[1]} vs {features[2]}")
    ax4.set_xlabel(features[1])
    ax4.set_ylabel(features[2])

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title)
    plt.show()


def transformed_space_plot(transformed_data, k, variance):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 2D plot
    ax[0].scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.7, edgecolor='k')
    ax[0].set_title("2D Projection (Principal Component 1 vs Principal Component 2)")
    ax[0].set_xlabel("Principal Component 1")
    ax[0].set_ylabel("Principal Component 2")
    ax[0].grid(True)

    # Subplot 2: 3D projection using the first three principal components
    if k >= 3:
        # 3D plot
        ax[1] = fig.add_subplot(122, projection='3d')
        ax[1].scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.7, edgecolor='k')
        ax[1].set_title("3D Projection (Principal Components 1, 2, and 3)")
        ax[1].set_xlabel("Principal Component 1")
        ax[1].set_ylabel("Principal Component 2")
        ax[1].set_zlabel("Principal Component 3")

    plt.suptitle(f"Transformed Dataset Components with {variance}% variance")
    plt.tight_layout()
    plt.show()


def plot_pca_metrics(captured_variances, reconstruction_errors, explained_variance_ratios):
    # Set up the bar plot
    x = np.arange(len(captured_variances))  # The label locations
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots for Reconstruction Error and Explained Variance
    bars1 = ax.bar(
        x - width / 2,
        reconstruction_errors,
        width,
        label="Reconstruction Error",
        color="dodgerblue",
        alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2,
        explained_variance_ratios,
        width,
        label="Explained Variance Ratio",
        color="royalblue",
        alpha=0.9
    )

    # Add text for labels, title, and axes
    ax.set_xlabel("Number of components")
    ax.set_title("Reconstruction Error and Explained Variance Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in captured_variances])
    ax.legend()

    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom")

    add_bar_labels(bars1)
    add_bar_labels(bars2)

    plt.tight_layout()
    plt.show()


def comparison_pca_plot(n_components_list, reconstruction_errors, explained_variance_ratios, times):
    methods = ["PCA", "Sklearn PCA", "Sklearn IPCA"]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Reconstruction Error
    x = np.arange(len(n_components_list))
    bar_width = 0.2
    for i, method in enumerate(methods):
        axs[0].bar(
            x + i * bar_width,
            reconstruction_errors[method],
            bar_width,
            label=method,
            color=["lightblue", "cornflowerblue", "royalblue"][i]  # Blue tones
        )
    axs[0].set_xticks(x + bar_width)
    axs[0].set_xticklabels(n_components_list)
    axs[0].set_title("Reconstruction Error")
    axs[0].set_xlabel("Number of Components")
    axs[0].set_ylabel("Reconstruction Error")
    axs[0].legend()

    # Explained Variance Ratio
    for i, method in enumerate(methods):
        axs[1].bar(
            x + i * bar_width,
            explained_variance_ratios[method],
            bar_width,
            label=method,
            color=["lightblue", "cornflowerblue", "royalblue"][i]  # Blue tones
        )
    axs[1].set_xticks(x + bar_width)
    axs[1].set_xticklabels(n_components_list)
    axs[1].set_title("Explained Variance Ratio")
    axs[1].set_xlabel("Number of Components")
    axs[1].set_ylabel("Explained Variance Ratio")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_comparison_varying_components(results, n_components_values, dataset_name, clustering_method):
    metrics = ['Silhouette', 'Calinski', 'Rand score', 'Jaccard Score']
    methods = list(results.keys())
    bar_width = 0.2
    method_colors = ['lightblue', 'cornflowerblue', 'royalblue', 'navy']

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharey=False)
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, (method, color) in enumerate(zip(methods, method_colors)):
            values = results[method][:, i]
            x_positions = np.arange(len(n_components_values)) + j * bar_width
            ax.bar(
                x_positions,
                values,
                width=bar_width,
                label=method,
                color=color,
                alpha=0.8
            )
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(n_components_values)) + bar_width)
        ax.set_xticklabels([f"{n}" for n in n_components_values])
        ax.set_xlabel('Number of PCA Components')
        ax.legend()

    plt.suptitle(f"Clustering with {clustering_method} Comparison by PCA Components for {dataset_name}")
    plt.tight_layout()
    plt.show()

