import numpy as np
import math
import time
from sklearn_relief import ReliefF
from exporter import load_predictions, export_predictions


class KNNClassifier:
    def __init__(self, dataset_name, X_train, y_train, k, r, weighting_technique, numerical_columns, voting_schema, distance_metric):
        self.k = k
        self.r = r
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.y_train = y_train
        self.numerical_columns = numerical_columns
        self.voting_schema = voting_schema
        self.distance_metric = distance_metric
        self.feature_weights = self.compute_weights(weighting_technique)


    def compute_weights(self, weighting_technique):
        if weighting_technique == 'relief':
            feature_weights = self.weights_relief(self.X_train, self.y_train)
        elif weighting_technique == 'correlation':
            feature_weights = self.weights_corr()
        else:
            feature_weights = np.ones(self.X_train.shape[1])
        return feature_weights


    def weights_relief(self, X_train, y_train):
        r = ReliefF(n_features=5)
        r.fit(X_train.values, y_train.values)
        return r.w_


    def weights_corr(self):
        corr_target = self.X_train.apply(lambda x: x.corr(self.y_train))
        corr_target_abs = corr_target.abs()
        corr_target_norm = corr_target_abs / corr_target_abs.sum()
        return corr_target_norm


    def minkowski_mixed_distance(self, x, y): # Manhattan: r=1, Euclidean: r=2
        minkowski_distance = 0
        for i in range(len(x)):
            col = self.X_train.columns[i]
            if col in self.numerical_columns:
                minkowski_distance += self.feature_weights[i]*abs(x[i]-y[i])**self.r
            else:
                minkowski_distance += self.feature_weights[i]*int(x[i] == y[i])
        return minkowski_distance**(1/self.r)


    def gower_distance(self, x, y):
        # Compute numerical feature ranges from the train set
        ranges = dict.fromkeys(self.numerical_columns)
        for num_col in self.numerical_columns:
            ranges[num_col] = max(self.X_train[num_col])-min(self.X_train[num_col])
        # Compute Gower distance
        gower_distance = 0
        for i in range(len(x)):
            col = self.X_train.columns[i]
            if col in self.numerical_columns:
                gower_distance += self.feature_weights[i]*(1-abs(x[i]-y[i]))/ranges[col]
            else:
                gower_distance += self.feature_weights[i]*int(x[i] == y[i])
        return gower_distance


    def voting(self, k_nearest_labels, k_nearest_distances):
        # Compute the most voted class depending on the voting schema
        if self.voting_schema == 'inverse_distance_weighted':
            labels = set(self.y_train)
            votes = dict(zip(list(labels), [0] * len(labels)))
            for lbl in labels:
                for k in range(len(k_nearest_labels)):
                    votes[lbl] += (1 / k_nearest_distances[k]) * int(lbl == k_nearest_labels[k])
            majority_label = max(votes, key=votes.get)
        elif self.voting_schema == 'voting_sheppard_work':
            labels = set(self.y_train)
            votes = dict(zip(list(labels), [0] * len(labels)))
            for lbl in labels:
                for k in range(len(k_nearest_labels)):
                    votes[lbl] = math.exp(-k_nearest_distances[k]) * int(lbl == k_nearest_labels[k])
            majority_label = max(votes, key=votes.get)
        else: # majority
            majority_label = max(set(k_nearest_labels), key=k_nearest_labels.count)

        return majority_label


    def predict(self, X_test):
        # Load the predictions if this model has already been executed
        #predictions = load_predictions(self.dataset_name, "knn", self.k, self.r)
        #if predictions:
        #    print("predictions loaded")
        #    return predictions
        predictions = []
        for i, test_point in X_test.iterrows():
            # Compute the distance between X_train and each point in X_test
            if self.distance_metric == 'gower':
                distances = [self.gower_distance(test_point, train_point) for _, train_point in self.X_train.iterrows()]
            else: # minkowski
                distances = [self.minkowski_mixed_distance(test_point, train_point) for _, train_point in
                             self.X_train.iterrows()]
            #k_indices = self.kneighbors(self.X_train, test_point, self.k)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train.iloc[k_i] for k_i in k_indices]
            k_nearest_distances = [distances[k_i] for k_i in k_indices]

            pred = self.voting(k_nearest_labels, k_nearest_distances)
            predictions.append(pred)
            # export_predictions(self.dataset_name, "knn", self.k, self.r, predictions)

        return predictions

    def kneighbors(self, X, x, n_neighbors):
        # Compute the distance between x and each point in X
        if self.distance_metric == 'gower':
            distances = [self.gower_distance(x, X_point) for _, X_point in X.iterrows()]
        else: # minkowski
            distances = [self.minkowski_mixed_distance(x, X_point) for _, X_point in
                         X.iterrows()]
        return np.argsort(distances)[:n_neighbors]




    def evaluate(self, X_test, y_test):
        start_time = time.time()
        y_pred = self.predict(X_test)
        end_time = time.time()

        accuracy = (y_pred == y_test).sum() / len(y_pred)

        return {
            'accuracy': float(accuracy),
            'consultation_time': end_time - start_time
        }