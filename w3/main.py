import algorithms.optics as optics
import algorithms.spectralclustering as spectralclustering
import algorithms.kmeans as kmeans
import algorithms.fuzzy as fuzzy
import warnings

from algorithms.evaluation_tools import plot_clusters, plot_reachability_distances
from algorithms.improved_kmeans import improved_kmeans_start
from preprocessing.preprocessor import preprocess_data

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    actual_k = {'hepatitis': 2, 'mx': 2, 'breast': 2}
    k_values = list(range(2, 7))
    print("\n--------------------------------------------------")
    print("\tWork 3 - Introduction to Machine Learning")
    print("--------------------------------------------------\n")
    while True:
        dataset_name = input("Type the name dataset to be analysed ('hepatitis', 'mx' or 'breast'): ")
        if dataset_name not in ["hepatitis", "mx", "breast"]:
            print("Invalid input format. Please try again.")
        else:
            # Obtain preprocessor and ideal k with known number of classes
            preprocessor = preprocess_data(dataset_name)
            print(preprocessor.original_numerical_columns)
            while True:
                print("\nType the number of the mode to be executed:\n")
                print("\t1) Optics Clustering\n\t2) Spectral Clustering")
                print("\t3) K-Means")
                print("\t4) Improved K-Means")
                print("\t5) Fuzzy Clustering \n\t6) Exit")
                mode = input("\nChoose a mode: ")
                if mode == "1":
                    # OPTICS
                    optics.run_internal_analysis(preprocessor.X, dataset_name, preprocessor.original_numerical_columns)
                    if dataset_name == 'hepatitis':
                        optics.run_external_analysis(preprocessor.X, preprocessor.y, dataset_name)
                    else:
                        print("External analysis was not performed on this dataset as the results of the internal analysis were too unsatisfactory")
                elif mode == "2":
                    # Spectral Clustering
                    spectralclustering.run_internal_analysis(preprocessor.X, dataset_name, k_values, preprocessor.original_numerical_columns)
                    #spectralclustering.run_test(preprocessor.X, dataset_name, k_values, preprocessor.original_numerical_columns)
                    spectralclustering.run_external_analysis(preprocessor.X, preprocessor.y)
                    # Individual clustering w/ optics
                    k, affinity, n_neighbors, eigen_solver, assign_labels = spectralclustering.get_spectral_input()
                    spec_cl = spectralclustering.SpectralClusteringAlgorithm(k, affinity, n_neighbors, eigen_solver, assign_labels)
                    labels_pred = spec_cl.fit_predict(preprocessor.X)
                    plot_clusters(preprocessor.X.to_numpy(), dataset_name, spec_cl.clusters)
                elif mode == "3":
                    # Plot performance with different k values
                    kmeans.run_internal_analysis(preprocessor.X.to_numpy(), preprocessor.y, dataset_name, k_values)
                    #kmeans.run_external_analysis(preprocessor.X.to_numpy(), preprocessor.y, dataset_name)
                    # Individual clustering w/ kmeans
                    k, distance = kmeans.get_kmeans_input()
                    km = kmeans.KMeans(k, distance)
                    km.fit_predict(preprocessor.X.to_numpy())
                    plot_clusters(preprocessor.X.to_numpy(), dataset_name, km.clusters, km.centroids)
                elif mode == "4":
                    improved_kmeans_start(preprocessor.X.to_numpy(), preprocessor.y, actual_k[dataset_name], dataset_name)
                elif mode == "5":
                    fuzzy.run_internal_analysis(preprocessor.X, dataset_name, k_values)
                    fuzzy.run_external_analysis(preprocessor.X, preprocessor.y, dataset_name)
                elif mode == "6":
                    print("Exiting the program. Goodbye!")
                    exit()
                else:
                    print("Invalid input. Please enter a number between 1 and 6.")