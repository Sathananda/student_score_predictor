#!/usr/bin/env python
# coding: utf-8
"""
Student Exam Score Predictor
Goal: Predict a Student's exam score based on the number of hours they studied using Linear Regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def create_dataset():
    """Creating mock data: Hours studied Vs Exam score (out of 100)"""
    hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    scores = np.array([15, 28, 40, 42, 55, 65, 78, 82, 91, 98])
    df = pd.DataFrame({'Hours_Studied': hours.flatten(), 'Exam_Score': scores})
    return df, hours, scores


def visualize_raw_data(df):
    """Visualizing the raw data"""
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Hours_Studied'], df['Exam_Score'], c='blue', marker='o')
    plt.title('Hours Studied Vs. Exam Score')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.grid(True)
    plt.show()


def train_simple_model(hours, scores):
    """Initialize and Train the simple model"""
    model = LinearRegression()
    model.fit(hours, scores)
    
    print("="*50)
    print(f"Simple Model Training Complete!")
    print(f"Slope (w/Coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept (b): {model.intercept_:.2f}")
    print("="*50)
    
    return model


def visualize_simple_model(hours, scores, model):
    """Plot the original data points with regression line"""
    predicted_score = model.predict(hours)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(hours, scores, c='blue', marker='x', label='Actual Scores', s=100)
    plt.plot(hours, predicted_score, c='red', linewidth=2, label='Model Prediction Line')
    plt.title("Hours Studied Vs. Exam Score (With fit line)")
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_multifeature_dataset():
    """Create dataset with multiple features: Hours Studied and Attendance Rate"""
    # New Data: [Hours Studied, Attendance Rate %]
    X_multi = np.array([
        [1, 50], [2, 60], [3, 75], [4, 70], [5, 85],
        [6, 80], [7, 90], [8, 95], [9, 98], [10, 100]
    ])
    scores = np.array([15, 28, 40, 42, 55, 65, 78, 82, 91, 98])
    
    df_multi = pd.DataFrame({
        'Hours_Studied': X_multi[:, 0],
        'Attendance(%)': X_multi[:, 1],
        'Scores_Obtained': scores
    }, index=range(1, 11))
    
    return df_multi, X_multi, scores


def train_multifeature_model(X_multi, scores):
    """Train a model with multiple features"""
    model_multi = LinearRegression()
    model_multi.fit(X_multi, scores)
    
    print("\n" + "="*50)
    print(f"Multi-Feature Model Training Complete!")
    print(f"Coefficients: {model_multi.coef_}")
    print(f"Intercept: {model_multi.intercept_:.2f}")
    print("="*50)
    
    return model_multi


def visualize_multifeature_model(df, X_multi, scores, model_multi):
    """Visualize multi-feature model predictions"""
    predict_score_multi = model_multi.predict(X_multi)
    
    # Set up a side-by-side graph for the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hours Studied Vs. Scores
    ax1.scatter(df['Hours_Studied'], df['Scores_Obtained'], c='blue', marker='o', s=80, label='Actual Scores')
    ax1.scatter(df['Hours_Studied'], predict_score_multi, c='red', marker='x', s=50, label='Predicted Model')
    ax1.set_title('Hours Studied Vs. Exam Scores')
    ax1.set_xlabel('Hours Studied')
    ax1.set_ylabel('Exam Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Attendance(%) Vs. Scores
    ax2.scatter(df['Attendance(%)'], df['Scores_Obtained'], c='blue', marker='o', s=80, label='Actual Scores')
    ax2.scatter(df['Attendance(%)'], predict_score_multi, c='red', marker='x', s=50, label='Predicted Model')
    ax2.set_title('Attendance(%) Vs. Exam Scores')
    ax2.set_xlabel('Attendance(%)')
    ax2.set_ylabel('Exam Scores')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def make_predictions(model, hours_value):
    """Make a prediction for new input"""
    prediction = model.predict([[hours_value]])
    return prediction[0]


if __name__ == "__main__":
    print("\n" + "="*50)
    print("STUDENT EXAM SCORE PREDICTOR")
    print("="*50 + "\n")
    
    # Simple Model: Hours Studied Only
    print("\n>>> Creating Dataset (Hours Studied vs Exam Score)...")
    df, hours, scores = create_dataset()
    print(df)
    
    print("\n>>> Visualizing Raw Data...")
    visualize_raw_data(df)
    
    print("\n>>> Training Simple Model (Hours Studied)...")
    model = train_simple_model(hours, scores)
    
    print("\n>>> Visualizing Simple Model Predictions...")
    visualize_simple_model(hours, scores, model)
    
    # Multi-feature Model: Hours Studied + Attendance
    print("\n>>> Creating Multi-Feature Dataset...")
    df_multi, X_multi, scores_multi = create_multifeature_dataset()
    print(df_multi)
    
    print("\n>>> Training Multi-Feature Model...")
    model_multi = train_multifeature_model(X_multi, scores_multi)
    
    print("\n>>> Visualizing Multi-Feature Model Predictions...")
    visualize_multifeature_model(df_multi, X_multi, scores_multi, model_multi)
    
    # Example Predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    test_hours = 5.5
    prediction = make_predictions(model, test_hours)
    print(f"\nFor {test_hours} hours studied, predicted score: {prediction:.2f}")
    
    print("\n" + "="*50)
    print("Project Complete!")
    print("="*50 + "\n")
