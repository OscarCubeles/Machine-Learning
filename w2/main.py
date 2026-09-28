from preprocessing.dataPreprocessing import DataPreprocessor
from program import Program
from analysis import Analysis
import warnings

from preprocessing.hepatitisPreprocessor import HepatitisPreprocessor
from preprocessing.nurseryPreprocessor import NurseryPreprocessor
from preprocessing.mxPreprocessor import MxPreprocessor

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("\n--------------------------------------------------")
    print("\tWork 2 - Introduction to Machine Learning")
    print("--------------------------------------------------\n")
    while True:
        dataset_name = input("\tType the dataset (hepatitis or mx):")
        if dataset_name not in ["hepatitis", "mx"]:
            print("Invalid input format. Please try again.")
        else:
            print("\n\t1) Preprocessing\n\t2) Algorithm analysis\n\t3) Algorithm evaluation")
            while True:
                mode = input("Choose a mode:")
                if mode=="1":
                    for fold in range(10):
                        # Preprocess and save one fold data
                        if dataset_name == "nursery":
                            preprocessor = NurseryPreprocessor(dataset_name, fold)
                        elif dataset_name == "hepatitis":
                            preprocessor = HepatitisPreprocessor(dataset_name, fold)
                        elif dataset_name == "mx":
                            preprocessor = MxPreprocessor(dataset_name, fold)
                        else:
                            raise ValueError("Unknown dataset. Please specify either 'nursery' or 'hepatitis'.")
                        preprocessor.preprocess_1_fold()
                        preprocessor.save_1_fold_data()
                    exit
                elif mode=="2":
                    analysis = Analysis(dataset_name)
                    analysis.start()

                elif mode=="3":
                    program = Program(dataset_name)
                    program.start()
                    exit

                else:
                    print("Invalid input. Please try again.")
