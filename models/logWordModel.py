from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


saved_model_name = "./saved_models/scaler_params.pkl"
saved_scaler_name = "./saved_models/model_params.pkl"


class LogWordModel:
    def __init__(self):
        self.logModel = LogisticRegression()
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.load_model(saved_model_name, saved_scaler_name)

    def load_model(self, scaler_file, model_file):
        scaler_params = joblib.load(scaler_file)
        self.scaler.mean_ = scaler_params['mean']
        self.scaler.scale_ = scaler_params['scale']

        model_params = joblib.load(model_file)
        self.logModel.intercept_ = model_params['intercept']
        self.logModel.coef_ = model_params['coefficients']
        self.best_threshold = model_params['threshold']
        self.logModel.classes_ = np.array([0, 1])

    def predict_probability(self, input_x):
        predicted_probability = self.logModel.predict_proba(input_x)[:, 1]

        return predicted_probability

    def predict_class(self, input_x):
        input_x = self.scaler.transform(input_x)
        predicted_probability = self.predict_probability(input_x)
        predicted_y = (predicted_probability >= self.best_threshold).astype(int)

        return np.column_stack((predicted_probability, predicted_y))
