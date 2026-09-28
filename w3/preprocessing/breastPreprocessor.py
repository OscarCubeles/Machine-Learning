from scipy.io.arff import loadarff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class BreastPreprocessor:
    def __init__(self):
        # Load data
        data, _ = loadarff("data/datasets/breast-w.arff")
        self.df = pd.DataFrame(data)
        self.label_columns = 'Class'
        self.df[self.label_columns] = self.df[self.label_columns].str.decode('utf-8').map({'benign': 0, 'malignant': 1})
        self.y = self.df[self.label_columns]
        self.X = self._drop_class()

        # Drop rows with too many NaNs
        self._drop_rows_with_threshold()

        # Preprocess and normalize the data
        self._process_numerical_data()
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

    def _normalize_data(self):
        scaler = MinMaxScaler()
        self.X[self.original_numerical_columns] = scaler.fit_transform(self.X[self.original_numerical_columns])

    def _drop_class(self):
        return self.df.drop(columns=[self.label_columns])