from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from algorithms.knnClassifier import KNNClassifier


class DataPreprocessor:
    def __init__(self, dataset_name, fold):
        self.dataset_name = dataset_name
        self.fold = fold
        self.df_train, self.df_test = self.load_1_fold_data()
        self.label_columns = None


    def load_1_fold_data(self):
        file_path_test = f'datasets/datasetsCBR/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.test.arff'
        file_path_train = f'datasets/datasetsCBR/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.train.arff'

        data_test, _ = loadarff(file_path_test)
        data_train, _ = loadarff(file_path_train)

        self.df_train = pd.DataFrame(data_train)
        self.df_test = pd.DataFrame(data_test)

        return self.df_train, self.df_test


    def get_processed_data(self, load_data=False):
        if load_data:
            file_path_test = f'datasets/datasetsPreprocessed/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.test.csv'
            file_path_train = f'datasets/datasetsPreprocessed/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.train.csv'
            self.df_train = pd.read_csv(file_path_train)
            self.df_test = pd.read_csv(file_path_test)

        X_train = self.df_train.drop(columns=self.label_columns)
        y_train = self.df_train[self.label_columns]
        X_test = self.df_test.drop(columns=self.label_columns)
        y_test = self.df_test[self.label_columns]

        return X_train, y_train, X_test, y_test


    def save_1_fold_data(self):
        file_path_test = f'datasets/datasetsPreprocessed/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.test.csv'
        file_path_train = f'datasets/datasetsPreprocessed/{self.dataset_name}/{self.dataset_name}.fold.{self.fold:06d}.train.csv'
        self.df_train.to_csv(file_path_train, index=False)
        self.df_test.to_csv(file_path_test, index=False)


    # condensed reduction technique
    def rnn(self):
        X_train = self.df_train.drop(columns=self.label_columns)
        y_train = self.df_train[self.label_columns]
        # Start with one random point from each class
        indices_to_remove = []

        # Iterate over each instance in the dataset
        for i in range(len(X_train)):
            # Temporarily remove the i-th instance
            indices = indices_to_remove + [i]
            temp_prototypes = X_train.drop(indices)

            # Check if the classification of the rest of the instances changes
            if temp_prototypes.size == 0:  # Skip if all prototypes are removed
                continue

            # Calculate distances from the remaining prototypes. rows = X, columns = temp_prototypes
            distances = cdist(X_train, temp_prototypes, metric='cityblock')

            # index of the nearest prototype for each x in X_train
            nearest_indices = np.argmin(distances, axis=1)
            predicted_labels = np.array([y_train.loc[nearest_indices[j]] for j in range(len(y_train))])

            # If all instances still classify correctly, mark this instance for removal
            if np.all(predicted_labels == y_train):
                indices_to_remove.append(i)

        # Return the reduced dataset
        indices_to_keep = [i for i in range(len(X_train)) if i not in indices_to_remove]

        y_train = np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        # Concatenate X_train and y_train horizontally
        self.df_train = pd.DataFrame(np.hstack((X_train.iloc[indices_to_keep], y_train[indices_to_keep])), columns=self.df_train.columns)


    # edited reduction technique
    def renn(self, user_input, numerical_columns):
        X_train = self.df_train.drop(columns=self.label_columns)
        y_train = self.df_train[self.label_columns]
        weighting_technique, distance, k, r, voting_schema = user_input.split('-')

        for j in range(2):
            knn = KNNClassifier(
                dataset_name=self.dataset_name,
                X_train=X_train,
                y_train=y_train,
                k=int(k),
                r=int(r),
                weighting_technique=weighting_technique,
                numerical_columns=numerical_columns,
                voting_schema=voting_schema,
                distance_metric=distance
            )

            # Predict the class of each instance using the k-nearest neighbors
            y_pred = knn.predict(X_train)

            # Find instances where the prediction matches the true class
            mask = [i for i, (a, b) in enumerate(zip(y_pred, y_train.tolist())) if a == b]

            # If no more instances are removed, stop the process
            if len(mask) == X_train.shape[0]:
                break

            # Keep only the correctly classified instances
            X_train = X_train.iloc[mask].reset_index(drop=True)
            y_train = y_train.iloc[mask].reset_index(drop=True)

        y_train = np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        # Concatenate X_train and y_train horizontally
        self.df_train = pd.DataFrame(np.hstack((X_train, y_train)),
                                     columns=self.df_train.columns)


    def hybrid(self, user_input, numerical_columns):
        # Apply RENN to remove noisy instances
        self.renn(user_input, numerical_columns)
        X_train_filtered = self.df_train.drop(columns=self.label_columns)
        y_train_filtered = self.df_train[self.label_columns]

        weighting_technique, distance, k, r, voting_schema = user_input.split('-')

        # List to keep track of which instances to keep
        keep_indices = np.arange(len(X_train_filtered))

        # Try to remove each instance
        for i in range(len(X_train_filtered)):
            # Initialize KNN classifier
            knn = KNNClassifier(
                dataset_name=self.dataset_name,
                X_train=X_train_filtered,
                y_train=y_train_filtered,
                k=int(k),
                r=int(r),
                weighting_technique=weighting_technique,
                numerical_columns=numerical_columns,
                voting_schema=voting_schema,
                distance_metric=distance
            )

            # Temporarily remove instance i
            temp_indices = np.delete(keep_indices, i)

            # Train KNN on the remaining instances
            y_train_pred = knn.predict(X_train_filtered.iloc[temp_indices])

            # Predict the class labels for the removed instance's neighbors
            neighbors = knn.kneighbors(X_train_filtered, X_train_filtered.iloc[i], n_neighbors=3)

            # Check if neighbors are still correctly classified without instance i
            y_train_pred = np.array(y_train_pred)
            if np.all(y_train_filtered.iloc[neighbors] == y_train_pred[neighbors]):
                # If neighbors are correctly classified, permanently remove instance i
                keep_indices = np.delete(keep_indices, i)

        X_train_filtered = X_train_filtered.iloc[keep_indices].reset_index(drop=True)
        y_train_filtered = y_train_filtered.iloc[keep_indices].reset_index(drop=True)
        # Concatenate X_train and y_train horizontally
        self.df_train = pd.DataFrame(np.hstack((X_train_filtered, y_train_filtered)),
                                     columns=self.df_train.columns)
