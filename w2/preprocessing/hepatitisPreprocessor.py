import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from preprocessing.dataPreprocessing import DataPreprocessor


class HepatitisPreprocessor(DataPreprocessor):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name, fold)
        self.label_columns = 'Class'
        self.original_numerical_columns = []

    def preprocess_1_fold(self):
        self._drop_rows_with_threshold()
        self._process_numerical_data()
        self._process_categorical_data()
        self._normalize_data()

    def _drop_rows_with_threshold(self):
        # Drop rows with more NaNs than the threshold defined
        threshold = int(0.3 * self.df_train.shape[1])
        self.df_train.dropna(thresh=self.df_train.shape[1] - threshold, inplace=True)
        self.df_test.dropna(thresh=self.df_test.shape[1] - threshold, inplace=True)

    def _process_numerical_data(self):
        self.original_numerical_columns = [col for col in self.df_train.columns if
                                           col not in self.df_train.select_dtypes(
                                               include=['object', 'category']).columns.tolist()]

        for col in self.original_numerical_columns:
            mean = self.df_train[col].mean()
            self.df_train[col].fillna(mean, inplace=True)
            self.df_test[col].fillna(mean, inplace=True)

    def _process_categorical_data(self):
        categorical_columns = self.df_train.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_columns:
            self.df_train[col] = self.df_train[col].str.decode('utf-8')
            self.df_test[col] = self.df_test[col].str.decode('utf-8')

        categorical_to_numerical = {"yes": 1, "no": 0, "male": 1, "female": 0, "LIVE" : 1, "DIE": 0, " ?": np.nan}

        for col in categorical_columns:
            self.df_train[col] = self.df_train[col].map(categorical_to_numerical)
            self.df_test[col] = self.df_test[col].map(categorical_to_numerical)

            mode = self.df_train[col].mode()[0]
            self.df_train[col].fillna(mode, inplace=True)
            self.df_test[col].fillna(mode, inplace=True)

    def _normalize_data(self):
        normalizer = Normalizer()

        self.df_train[self.original_numerical_columns] = normalizer.fit_transform(self.df_train[self.original_numerical_columns])
        self.df_test[self.original_numerical_columns] = normalizer.fit_transform(self.df_test[self.original_numerical_columns])

