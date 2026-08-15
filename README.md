
# Medical Insurance Cost Prediction using Multiple Linear Regression

## Project Overview

This project predicts medical insurance charges based on customer information such as age, BMI, gender, smoking status, number of children, and region.

The project demonstrates a complete Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## Features

- Data Preprocessing
- Missing Value Handling
- Outlier Detection
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Multiple Linear Regression
- Gradient Descent (from Scratch)
- Model Evaluation
- Data Visualization

---

## Dataset

-age-Age of customer
-sex-Male/Female
-bmi-Body Mass Index
-children-Number of children
-smoker-Smoking status
-region-Residential region
-charges-Insurance charges

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Workflow

1. Load Dataset
2. Handle Missing Values
3. Remove Duplicates
4. Handle Inconsistent Data
5. Encode Categorical Variables
6. Detect and Handle Outliers
7. Split Dataset into Training and Testing Sets
8. Perform Feature Scaling
9. Train Multiple Linear Regression Model
10. Predict Insurance Charges
11. Evaluate Model Performance
12. Visualize Results

---

## Model Evaluation

The model was evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Results

MSE : 113015619.49
MAE : 2688.37
R2_Score : 0.45
RMSE : 10630.88
Accuracy : 44.72 %

---

## Visualizations

### Actual vs Predicted Charges

![Actual vs Predicted](screenshots/Actual%20vs%20Predicated.png)

### Feature Coefficients

![Feature Coefficients](screenshots/model%20co-efficient.png)

---

## Installation

### Clone the Repository
git clone https://github.com/rp1794/Medical-Insurance-Cost-Prediction.git


### Install Required Libraries
pip install -r requirements.txt


---

## Sample Prediction

### Input

- Age: 35
- Gender: Male
- BMI: 27.5
- Children: 2
- Smoker: No
- Region: Northeast

### Predicted Insurance Charge

₹18,520.46

---

## Future Improvements

- Deploy the model using Flask
- Compare Multiple Linear Regression with Random Forest
- Compare Multiple Linear Regression with XGBoost
- Perform Hyperparameter Tuning
- Build an Interactive Web Application

---

## Author

Riya S. Patel

M.Sc. Artificial Intelligence & Machine Learning