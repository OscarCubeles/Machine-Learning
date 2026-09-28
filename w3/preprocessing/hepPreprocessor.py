from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class HepPreprocessor:
    def __init__(self):
        # Load data
        data, _ = loadarff("data/datasets/hepatitis.arff")
        self.df = pd.DataFrame(data)
        self.label_columns = 'Class'
        self.y = self.df[self.label_columns]
        self.X = self._drop_class()

        # Drop rows with too many NaNs
        self._drop_rows_with_threshold()

        # Preprocess and normalize the data
        self._process_numerical_data()
        self._process_categorical_data()
        self._normalize_data()

    def _drop_rows_with_threshold(self):
        threshold = int(0.3 * self.X.shape[1])  # Allow up to 30% missing values in a row
        self.X.dropna(thresh=self.X.shape[1] - threshold, inplace=True)

    def _process_numerical_data(self):
        self.original_numerical_columns = [col for col in self.X.columns if
                                           col not in self.X.select_dtypes(
                                               include=['object', 'category']).columns]

        for col in self.original_numerical_columns:
            mean = self.X[col].mean()
            self.X[col].fillna(mean, inplace=True)

    def _process_categorical_data(self):
        categorical_columns = self.X.select_dtypes(include=['object']).columns.tolist()

        # Decode byte strings to UTF-8
        for col in categorical_columns:
            self.X[col] = self.X[col].str.decode('utf-8')
        self.y = self.y.str.decode('utf-8')

        # Mapping for categorical values
        categorical_to_numerical = {"yes": 1, "no": 0, "male": 1, "female": 0, "LIVE": 1, "DIE": 0, " ?": np.nan}

        for col in categorical_columns:
            self.X[col] = self.X[col].map(categorical_to_numerical)

            # Fill missing values with the mode
            mode = self.X[col].mode()[0]
            self.X[col].fillna(mode, inplace=True)

        self.y = self.y.map(categorical_to_numerical)

    def _normalize_data(self):
        scaler = MinMaxScaler()
        self.X[self.original_numerical_columns] = scaler.fit_transform(self.X[self.original_numerical_columns])

    def _drop_class(self):
        return self.df.drop(columns=['Class'])