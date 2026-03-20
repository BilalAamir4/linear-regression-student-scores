import pandas as pd
import numpy as np
import pickle

from preprocess import normalize_train
from model import train_linear_regression, predict
from visualize import plot_loss, plot_predictions, plot_feature_vs_target

# Load dataset
data = pd.read_csv("../data/student_scores.csv")

data = data.drop("student_id", axis=1)

X = data.drop("exam_score", axis=1).values
y = data["exam_score"].values

# Save original X for plotting
X_original = X.copy()

# Normalize
X, min_vals, max_vals = normalize_train(X)

# Train
weights, losses = train_linear_regression(X, y)

# Predictions on training data
y_pred = predict(X, weights)

# 📊 VISUALIZATIONS
plot_loss(losses)
plot_predictions(y, y_pred)

feature_names = ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]

for i, name in enumerate(feature_names):
    plot_feature_vs_target(X_original, y, i, name)

# Save model
model_data = {
    "weights": weights,
    "min_vals": min_vals,
    "max_vals": max_vals
}

with open("../model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model trained + visualizations saved in /results")