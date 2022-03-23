#Import
import joblib
import pandas as pd

#Load Data and Model
test_data = pd.read_csv("Data/test_samples.csv", sep=";")
cardio_predictor = joblib.load("Model/cardio_predictor")

#Predict values
predicted_values_for_deployed = cardio_predictor.predict(test_data)

predictions = pd.DataFrame(columns=["Probability class 1", "Probability class 2", "Prediction"])

