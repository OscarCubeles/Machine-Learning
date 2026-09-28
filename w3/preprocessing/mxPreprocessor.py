from scipy.io.arff import loadarff
import pandas as pd


class MxPreprocessor:
    def __init__(self):
        # Load the ARFF dataset
        data, _ = loadarff("data/datasets/mx.arff")
        self.label_columns = 'class'
        self.X = pd.DataFrame(data)
        self.y = self.X[self.label_columns]
        self.X = self._drop_class()

        # Preprocess the data
        self._decode()

    def _decode(self):
        self.original_numerical_columns = [col for col in self.X.columns if col != self.label_columns]

        for col in self.original_numerical_columns:
            # Decode bytes to strings and convert to integer type
            self.X[col] = self.X[col].str.decode('utf-8').astype(int)
        self.y = self.y.str.decode('utf-8').astype(int)

    def _drop_class(self):
        return self.X.drop(columns=[self.label_columns])