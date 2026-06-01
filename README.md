# Student Exam Score Predictor 

A Machine Learning project that predicts student exam scores based on study hours and attendance using **Linear Regression**.

## Project Overview

**Goal:** Predict a student's exam score based on the number of hours they studied using Linear Regression.

This project demonstrates:
- Simple Linear Regression (Single Feature)
- Multiple Linear Regression (Multiple Features)
- Data Visualization
- Model Training and Evaluation

---

## Features

### 1. **Simple Model** (Hours Studied)
- Input: Hours studied
- Output: Predicted exam score
- Learns the relationship between hours of study and exam scores

### 2. **Multi-Feature Model** (Hours Studied + Attendance)
- Input: Hours studied + Attendance percentage
- Output: Predicted exam score
- More accurate predictions with multiple features

---

## Dataset

The project uses mock data with 10 samples:
- **Hours Studied:** 1-10 hours
- **Attendance Rate:** 50-100%
- **Exam Scores:** 15-98 (out of 100)

---

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sathananda/student_score_predictor.git
   cd student_score_predictor
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the project
```bash
python main.py
```

This will:
1. Create the dataset
2. Visualize raw data
3. Train the simple model
4. Train the multi-feature model
5. Display predictions and visualizations

---

## Project Structure

```
student_score_predictor/
│
├── main.py                 # Main project file
├── requirements.txt        # Project dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
│
└── outputs/               # Generated visualizations (optional)
    ├── raw_data.png
    ├── simple_model.png
    └── multi_feature_model.png
```

---

## Results

### Simple Model (Hours Studied)
- **Slope (Coefficient):** 8.42
- **Intercept:** 4.77
- **Formula:** Score = 8.42 × Hours + 4.77

### Multi-Feature Model (Hours + Attendance)
- Uses both hours of study and attendance rate
- Provides more accurate predictions

### Example Predictions
- **5.5 hours studied:** ~48 score

---

## Dependencies

- **NumPy:** Numerical computations
- **Pandas:** Data manipulation and analysis
- **Scikit-learn:** Machine learning algorithms (Linear Regression)
- **Matplotlib:** Data visualization
- **Jupyter:** Interactive notebooks

---

## Key Concepts Covered

1. **Linear Regression:** Understanding the linear relationship between features and target
2. **Model Training:** Fitting the model to training data
3. **Predictions:** Making predictions on new data
4. **Visualization:** Plotting actual vs predicted values
5. **Multi-feature Models:** Using multiple features for better predictions

---

## Author

**Sathananda**

---

## License

This project is open source and available under the MIT License.

---

## Disclaimer

This is an educational project using mock data for demonstration purposes. Real-world predictions require actual datasets and proper validation techniques.

---

## Future Enhancements

- [ ] Add more features (previous scores, class participation, etc.)
- [ ] Implement cross-validation
- [ ] Add model evaluation metrics (R² score, RMSE)
- [ ] Create a web interface (Flask/Streamlit)
- [ ] Add data persistence (save/load models)
- [ ] Implement polynomial regression for comparison

---

## Contact & Support

For questions or suggestions, please open an issue on GitHub.

Happy Learning! 🚀
