import numpy as np
import pickle

from preprocess import normalize_test
from model import predict

# Load model
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
min_vals = model_data["min_vals"]
max_vals = model_data["max_vals"]

# 🎯 Take user input
hours = float(input("📘 Hours studied: "))
sleep = float(input("😴 Sleep hours: "))
attendance = float(input("🏫 Attendance (%): "))
previous = float(input("📊 Previous score: "))

# Create input array
new_data = np.array([[hours, sleep, attendance, previous]])

# Normalize using training values
new_data = normalize_test(new_data, min_vals, max_vals)

# Predict
prediction = predict(new_data, weights)

print("\n🎯 Predicted Exam Score:", round(prediction[0], 2))