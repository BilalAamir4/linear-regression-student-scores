#  Linear Regression from Scratch – Student Exam Score Predictor

This project implements Linear Regression from scratch (without using machine learning libraries) to predict student exam scores based on study-related features.

---

##  Objective

The goal is to predict exam scores using the following features:

- Hours Studied  
- Sleep Hours  
- Attendance Percentage  
- Previous Scores  

---

##  Algorithm Used

This project uses Multiple Linear Regression:

y = b₀ + b₁x₁ + b₂x₂ + b₃x₃ + b₄x₄

Where:
- y = exam score  
- x₁, x₂, x₃, x₄ = input features  

---

##  Key Concepts Implemented

- Linear Regression from scratch  
- Gradient Descent optimization  
- Mean Squared Error (MSE) loss  
- Feature Normalization (Min-Max Scaling)  
- Bias handling  
- Model saving & loading  
- Data visualization  

---

## 📁 Project Structure

Linear-Regression-Student-Scores/
│
├── data/
│   └── student_scores.csv
│
├── results/
│   ├── loss_curve.png
│   ├── predictions_vs_actual.png
│   └── *_vs_target.png
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── model.pkl
├── requirements.txt
└── README.md



##  How to Run

Step 1: Install dependencies  
Run the following command in your terminal:
pip install -r requirements.txt

Step 2: Train the model  
Run:
python src/train.py

This will:
- Train the model  
- Save model.pkl  
- Generate visualizations in the results folder  

Step 3: Make predictions  
Run:
python src/predict.py

Then enter values like:
Hours studied: 6  
Sleep hours: 7  
Attendance (%): 85  
Previous score: 78  

Output will be:
Predicted Exam Score: 81.34  



##  Visualizations

The project generates the following:

- Loss Curve: shows how the model improves during training  
- Predicted vs Actual: compares predicted scores with real values  
- Feature vs Target: shows relationship between each feature and exam score  

---

## ⚠️ Important Note

Normalization is applied using training data statistics and reused during prediction to ensure consistent and correct results.

---

##  Why This Project?

Most implementations rely on libraries like scikit-learn.  
This project builds everything from scratch to demonstrate a clear understanding of core machine learning concepts.

---

##  Author

Bilal Aamir

---
