import pandas as pd

from preprocessing.dataPreprocessing import DataPreprocessor


class NurseryPreprocessor(DataPreprocessor):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name, fold)
        self.label_columns = ['class_not_recom', 'class_priority', 'class_recommend', 'class_spec_prior',
                              'class_very_recom']
        self.original_numerical_columns = []


    def preprocess_1_fold(self):
        self._decode_and_one_hot_encode()


    def _decode_and_one_hot_encode(self):
        categorical_columns = self.df_train.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_columns:
            self.df_train[col] = self.df_train[col].str.decode('utf-8')
            self.df_test[col] = self.df_test[col].str.decode('utf-8')

        self.df_train = pd.get_dummies(self.df_train,
                                       columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'class']).astype(int)
        self.df_test = pd.get_dummies(self.df_test,
                                      columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'class']).astype(int)

        self.df_test['class_recommend'] = 0
        self.df_test = self.df_test[self.df_train.columns]
