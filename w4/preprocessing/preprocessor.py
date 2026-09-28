from preprocessing.breastPreprocessor import BreastPreprocessor
from preprocessing.hepPreprocessor import HepPreprocessor


def preprocess_data(dataset):
    if dataset == "hepatitis":
        preprocessor = HepPreprocessor()
    elif dataset == "breast":
        preprocessor = BreastPreprocessor()

    return preprocessor


def get_features(dataset):
    if dataset == "hepatitis":
        features = ['PROTIME', 'SGOT', 'AGE']
    elif dataset == "breast":
        features = ['Clump_Thickness', 'Mitoses', 'Bare_Nuclei']

    return features
