import warnings

from algorithms.pca_analysis import plot_comparison_pca_3d
from algorithms.reduced_clustering import input_kmeans_optics
from algorithms.visualization import input_visualization
from preprocessing.preprocessor import preprocess_data, get_features
from algorithms.pca_analysis import pca_comparison, pca_analysis

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("\n--------------------------------------------------")
    print("\tWork 4 - Introduction to Machine Learning")
    print("--------------------------------------------------\n")
    while True:
        dataset_name = input("Type the name dataset to be analysed ('hepatitis' or 'breast'): ")
        if dataset_name not in ["hepatitis", "breast"]:
            print("Invalid input format. Please try again.")
        else:
            # Obtain preprocessor and ideal k with known number of classes
            preprocessor = preprocess_data(dataset_name)
            features = get_features(dataset_name)
            while True:
                print("\nType the number of the mode to be executed:\n")
                print("\t1) PCA\n\t2) PCA Comparison")
                print("\t3) PCA + Clustering\n\t4) Visualization")
                print("\t3) Exit")
                mode = input("\nChoose a mode: ")
                if mode == "1":
                    pca_analysis(preprocessor.X, features)
                elif mode == "2":
                    pca_comparison(preprocessor.X, features)
                    plot_comparison_pca_3d(preprocessor.X, features, n_components=3)
                elif mode == "3":
                    input_kmeans_optics(preprocessor.X, preprocessor.y, features, dataset_name)
                elif mode == "4":
                    input_visualization(preprocessor.X, preprocessor.y, features, dataset_name)
                elif mode == "5":
                    print("Exiting the program. Goodbye!")
                    exit()


