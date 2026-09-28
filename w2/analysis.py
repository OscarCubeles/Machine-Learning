import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.hepatitisPreprocessor import HepatitisPreprocessor
from preprocessing.mxPreprocessor import MxPreprocessor
from algorithms.knnClassifier import KNNClassifier
from algorithms.svmAlgorithm import SVMAlgorithm

class Analysis():
    def __init__(self, dataset_name):
        self.exit_analysis = False
        self.dataset_name = dataset_name


    def start(self):
        while not self.exit_analysis:
            user_input = self.initial_view()
            self.process_input(user_input)


    def initial_view(self):
        while True:
            algorithm = input("Please choose an algorithm:")
            print("Type 'exit' to quit the program.\n")
            if algorithm.lower() == "exit":
                self.exit_program = True
                print("Exiting the program. Goodbye!")
                return

            if algorithm.lower() == "knn":
                self.KNN_analysis()
            elif algorithm.lower() == "svm":
                self.SVM_analysis()
            else:
                print("Invalid input. Please try again.")


    def KNN_analysis(self):
        # Define the parameters to be tested
        k_values = [1, 3, 5, 7]
        weighting_techniques = ['relief', 'correlation', 'none']  # none: tots els weights a 1
        distance_metrics = ['minkowski', 'gower']
        r_values = [1, 2]
        voting_schemas = ['inverse_distance_weighted', 'voting_sheppard_work', 'majority']

        results = pd.DataFrame(columns=['k', 'r', 'weighting_technique', 'distance_metric', 'voting_schema', 'accuracies', 'efficiencies', 'avg_accuracy', 'avg_efficiency'])

        folds_preprocessed = []
        for fold in range(10):
            if self.dataset_name == "hepatitis":
                preprocessor = HepatitisPreprocessor(self.dataset_name, fold)
            else:
                preprocessor = MxPreprocessor(self.dataset_name, fold)
            folds_preprocessed.append((preprocessor.get_processed_data(load_data=True)))

        # Perform 10-fold cross-validation on every combination of parameters
        for k in k_values:
            for weighting_technique in weighting_techniques:
                for voting_schema in voting_schemas:
                    for distance_metric in distance_metrics:
                        # Minkowski distance
                        if distance_metric == 'minkowski':
                            for r in r_values:
                                accuracy = []
                                efficiency = []
                                print(f"Starting 10-fold validation with\nk={k},\nr={r},\ndistance_metric={distance_metric},\nweighting_technique={weighting_technique},\nvoting_schema={voting_schema}...")
                                for i in tqdm(range(10)):
                                    X_train, y_train, X_test, y_test = folds_preprocessed[i]
                                    knn = KNNClassifier(
                                        dataset_name=self.dataset_name,
                                        X_train=X_train,
                                        y_train=y_train,
                                        k=k,
                                        r=r,
                                        weighting_technique=weighting_technique,
                                        numerical_columns=preprocessor.original_numerical_columns,
                                        voting_schema=voting_schema,
                                        distance_metric=distance_metric
                                    )

                                    performance = knn.evaluate(X_test, y_test)
                                    accuracy.append(round(performance['accuracy'],4))
                                    efficiency.append(round(performance['consultation_time'],4))

                                # Compute fold accuracy and efficiency averages
                                avg_accuracy = np.mean(accuracy)
                                avg_efficiency = np.mean(efficiency)

                                # Save data in DataFrame
                                results.loc[len(results)] = [k, r, weighting_technique, distance_metric, voting_schema,
                                                             accuracy, avg_accuracy, efficiency, avg_efficiency]
                        else:
                            # Gower distance (no r)
                            accuracy = []
                            efficiency = []

                            print(f"\nStarting 10-fold validation with\nk={k},\nr=None,\ndistance_metric={distance_metric},\nweighting_technique={weighting_technique},\nvoting_schema={voting_schema}...")
                            for i in tqdm(range(10)):
                                X_train, y_train, X_test, y_test = folds_preprocessed[i]
                                knn = KNNClassifier(
                                    dataset_name=self.dataset_name,
                                    X_train=X_train,
                                    y_train=y_train,
                                    k=k,
                                    r=None,
                                    weighting_technique=weighting_technique,
                                    numerical_columns=preprocessor.original_numerical_columns,
                                    voting_schema=voting_schema,
                                    distance_metric=distance_metric
                                )

                                performance = knn.evaluate(X_test, y_test)
                                accuracy.append(round(performance['accuracy'], 4))
                                efficiency.append(round(performance['consultation_time'],4))

                            # Compute fold accuracy and efficiency averages
                            avg_accuracy = np.mean(accuracy)
                            avg_efficiency = np.mean(efficiency)

                            # Save data in DataFrame
                            results.loc[len(results)] = [k, '-', weighting_technique, distance_metric, voting_schema,
                                                         accuracy, efficiency, round(avg_accuracy,4), round(avg_efficiency,4)]

        # Save results
        results.to_csv('results/knn_analysis.csv', index=False)

    def SVM_analysis(self):

        kernels = ['poly', 'rbf']
        results_poly = pd.DataFrame(columns=['C', 'degree', 'coef0', 'accuracies', 'efficiencies', 'avg_accuracy', 'avg_efficiency'])
        results_rbf = pd.DataFrame(columns=['C', 'gamma', 'accuracies', 'efficiencies', 'avg_accuracy', 'avg_efficiency'])

        folds_preprocessed = []
        for fold in range(10):
            if self.dataset_name == "hepatitis":
                preprocessor = HepatitisPreprocessor(self.dataset_name, fold)
            else:
                preprocessor = MxPreprocessor(self.dataset_name, fold)
            folds_preprocessed.append((preprocessor.get_processed_data(load_data=True)))

        for kernel in kernels:
            # Polinomic kernel
            if kernel == 'poly':
                for C in [0.001, 0.01, 0.1, 10, 50, 100]:
                    for degree in [3, 5, 7, 10]:
                        for coef0 in [0.1, 1, 5, 10]:

                            accuracies = []
                            efficiencies = []
                            print(f"Starting 10-fold validation with\nC={C},\ndegree={degree},\ncoef={coef0}...")
                            for i in tqdm(range(10)):
                                X_train, y_train, X_test, y_test = folds_preprocessed[i]
                                svm = SVMAlgorithm(kernel=kernel, C=C, degree=degree, coef0=coef0)
                                svm.fit(X_train, y_train)
                                performance = svm.evaluate(X_test, y_test)
                                accuracies.append(round(performance['accuracy'], 4))
                                efficiencies.append(round(performance['consultation_time'],4))

                            # Compute fold accuracy and efficiency averages
                            avg_accuracy = np.mean(accuracies)
                            avg_efficiency = np.mean(efficiencies)

                            # Save data in DataFrame
                            results_poly.loc[len(results_poly)] = [C, degree, coef0,
                                                                   accuracies, efficiencies, round(avg_accuracy,4), round(avg_efficiency,4)]
            else:
                # Gausian kernel
                for C in [0.001, 0.01, 0.1, 10, 50, 100]:
                    for gamma in ['auto', 1, 0.1, 0.01, 0.001]:

                        accuracies = []
                        efficiencies = []
                        print(f"Starting 10-fold validation with\nC={C},\ngamma={gamma}...")
                        for i in tqdm(range(10)):
                            X_train, y_train, X_test, y_test = folds_preprocessed[i]
                            svm = SVMAlgorithm(kernel=kernel, C=C, gamma=gamma)
                            svm.fit(X_train, y_train)
                            performance = svm.evaluate(X_test, y_test)
                            accuracies.append(round(performance['accuracy'],4))
                            efficiencies.append(round(performance['consultation_time'],4))

                        # Compute fold accuracy and efficiency averages
                        avg_accuracy = np.mean(accuracies)
                        avg_efficiency = np.mean(efficiencies)

                        # Save data in DataFrame
                        results_rbf.loc[len(results_rbf)] = [C, gamma, accuracies, efficiencies, round(avg_accuracy,4), round(avg_efficiency,4)]

        # Save results
        results_poly.to_csv(f'results/svm_poly_analysis_{self.dataset_name}.csv', index=False)
        results_rbf.to_csv(f'results/svm_rbf_analysis_{self.dataset_name}.csv', index=False)