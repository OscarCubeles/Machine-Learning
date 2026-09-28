import pandas as pd

from algorithms.knnClassifier import KNNClassifier
from preprocessing.hepatitisPreprocessor import HepatitisPreprocessor
from preprocessing.mxPreprocessor import MxPreprocessor

# Configurar Pandas para mostrar todas las columnas sin truncar
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.expand_frame_repr', False)  # Evita que el DataFrame se divida en varias líneas
pd.set_option('display.max_colwidth', None)  # Evita que las columnas con texto largo se trunquen


class Program:
    def __init__(self, dataset_name):
        self.exit_program = False
        self.dataset_name = dataset_name


    def start(self):
        while not self.exit_program:
            algorithm, user_input = self.initial_view()
            self.process_input(algorithm, user_input)


    def initial_view(self):
        print("\n--------------------------------------------------")
        print("\tWork 2 - Introduction to Machine Learning")
        print("--------------------------------------------------\n")
        while True:
            algorithm = input("Please choose an algorithm (knn or svm) or type 'exit' to exit:")
            if algorithm.lower() == "exit":
                self.exit_program = True
                print("Exiting the program. Goodbye!")
                return
            elif algorithm == "knn":
                print("\tPlease enter your command in the following format:")
                print("\t<weighting_technique>-<distance>-<k>-<r>-<voting_schema> (e.g: relief-gower-3-2-majority)")
                print("\tType 'exit' to quit the program.\n")

                while True:
                    user_input = input("Enter your command: ").strip()
                    print()
                    if user_input.lower() == "exit":
                        self.exit_program = True
                        print("Exiting the program. Goodbye!")
                        return

                    if self.validate_input_knn(user_input):
                        return algorithm, user_input
                    else:
                        print("Invalid input format. Please try again.")
            elif algorithm == "svm":
                # create code for svm
                return
            else:
                print("Invalid input format. Please try again.")


    def validate_input_knn(self, user_input):
        try:
            parts = user_input.split('-')
            if len(parts) != 5:
                return False

            weighting_technique, distance, k, r, voting_schema = parts
            if weighting_technique not in ["relief", "correlation"]:
                return False
            if voting_schema not in ["inverse_distance_weighted", "voting_sheppard_work", "majority"]:
                return False
            if distance not in ["gower", "minkowski"]:
                return False
            k = int(k)
            r = int(r)
            if k not in [1, 3, 5, 7]:
                return False
            if r not in [1, 2]:
                return False

        except ValueError:
            return False  # Return false if there's a conversion error

        return True  # Input is valid


    def process_input(self, algorithm, user_input):
        reduction_techniques = True
        fold = 0

        if user_input is None:  # Exit command detected
            return
        if algorithm == "knn":
            weighting_technique, distance, k, r, voting_schema = user_input.split('-')

            if self.dataset_name == "hepatitis":
                preprocessor = HepatitisPreprocessor(self.dataset_name, fold)
            else:
                preprocessor = MxPreprocessor(self.dataset_name, fold)

            X_train, y_train, X_test, y_test = preprocessor.get_processed_data(load_data=True)

            if reduction_techniques:
                reductionKNNAlgorithm(preprocessor, user_input, preprocessor.original_numerical_columns)

            # Apply knn classifier
            knn = KNNClassifier(
                dataset_name=self.dataset_name,
                X_train=X_train,
                y_train=y_train,
                k=int(k),
                r=int(r),
                weighting_technique=weighting_technique,
                numerical_columns=preprocessor.original_numerical_columns,
                voting_schema=voting_schema,
                distance_metric=distance
            )

            performance = knn.evaluate(X_test, y_test)
            print("Accuracy: ", round(performance['accuracy'], 4))
            print("Efficiency: ", round(performance['consultation_time'], 4))

            results = pd.DataFrame(columns=['k', 'r', 'weighting_technique', 'distance_metric', 'voting_schema', 'accuracy', 'efficiency'])
            exit
        elif algorithm == "svm":
            # codi svm
            return

def reductionKNNAlgorithm(preprocessor, user_input, original_numerical_columns):
    # condensed technique
    #preprocessor.rnn()

    # edited technique
    #preprocessor.renn(user_input, original_numerical_columns)

    #hybrid technique
    preprocessor.hybrid(user_input, original_numerical_columns)


