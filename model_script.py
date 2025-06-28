# model_script.py
import joblib
import json
import numpy as np

# Load label mapping
with open(r"D:\DAV Project\label_mapping.json") as f:
    label_mapping = json.load(f)
    label_reverse = {v: k for k, v in label_mapping.items()}

# Load the trained pipeline
loaded_model = joblib.load(r"D:\DAV Project\crop_recommender_rf.pkl")

# Predict function
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    sample = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = loaded_model["model"].predict(sample)[0]
    return label_reverse[int(prediction)]
