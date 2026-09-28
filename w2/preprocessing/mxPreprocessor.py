import pandas as pd

from preprocessing.dataPreprocessing import DataPreprocessor


class MxPreprocessor(DataPreprocessor):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name, fold)
        self.label_columns = ['class_0', 'class_1']
        self.original_numerical_columns = []

    def preprocess_1_fold(self):
        self._decode_and_one_hot_encode()

    def _decode_and_one_hot_encode(self):
        self.original_numerical_columns = [col for col in self.df_train.columns]

        for col in self.original_numerical_columns:
            self.df_train[col] = self.df_train[col].str.decode('utf-8').astype(int)
            self.df_test[col] = self.df_test[col].str.decode('utf-8').astype(int)

        self.df_train = pd.get_dummies(self.df_train, columns=['class']).astype(int)
        self.df_test = pd.get_dummies(self.df_test, columns=['class']).astype(int)




