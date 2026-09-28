from preprocessing.breastPreprocessor import BreastPreprocessor
from preprocessing.hepPreprocessor import HepPreprocessor
from preprocessing.mxPreprocessor import MxPreprocessor


def preprocess_data(dataset):
    if dataset == "hepatitis":
        preprocessor = HepPreprocessor()
    elif dataset == "mx":
        preprocessor = MxPreprocessor()
    elif dataset == "breast":
        preprocessor = BreastPreprocessor()

    return preprocessor