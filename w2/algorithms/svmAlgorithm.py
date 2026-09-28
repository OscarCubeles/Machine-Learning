from sklearn.svm import SVC
import joblib
import time

class SVMAlgorithm(SVC):
    def __init__(self, kernel, C, degree=3, coef0=0.0, gamma='scale'):
        # Initialize the parent class
        super().__init__(kernel=kernel, C=C, degree=degree, coef0=coef0, gamma=gamma)


    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
        return self


    def predict(self, X_test):
        return super().predict(X_test)


    def evaluate(self, X_test, y_test):
        start_time = time.time()
        y_pred = self.predict(X_test)
        end_time = time.time()

        accuracy = (y_pred == y_test).sum() / len(y_pred)

        return {
            'accuracy': float(accuracy),
            'consultation_time': end_time - start_time
        }

