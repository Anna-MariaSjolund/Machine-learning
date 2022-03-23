#Import packages
import joblib
import pandas as pd

#Load data and model
test_data = pd.read_csv("Data/test_samples.csv")
cardio_predictor = joblib.load("Model/cardio_predictor")

#Predict values
results = pd.DataFrame(cardio_predictor.predict_proba(test_data), columns=["probability class 0", "probability class 1"])
results["prediction"] = cardio_predictor.predict(test_data)

#Save the predicted values
results.to_csv("Data/prediction.csv", index=False)