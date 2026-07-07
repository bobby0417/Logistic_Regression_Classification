# Classification with Logistic Regression

## Project Overview

This project demonstrates how to build a Binary Classification model using Logistic Regression in Python. The Breast Cancer Wisconsin dataset from Scikit-learn is used to classify tumors as **Malignant** or **Benign**. The project also evaluates the model using different performance metrics and visualizations.

---

## Objective

- Build a Binary Classification model using Logistic Regression.
- Split the dataset into training and testing sets.
- Standardize the input features.
- Train the Logistic Regression model.
- Evaluate the model using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Visualize the results using Confusion Matrix and ROC Curve.

---

## Dataset

**Breast Cancer Wisconsin Dataset**

- Source: Scikit-learn
- Number of Samples: 569
- Number of Features: 30
- Target Classes:
  - Malignant (Cancerous)
  - Benign (Non-Cancerous)

---

## Technologies Used

- Python 3.12+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Structure

```
Logistic_Regression_Classification/
│── logistic_regression.py
│── README.md
│── requirements.txt
│── confusion_matrix.png
│── roc_curve.png
```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## How to Run

Open the project folder in VS Code.

Run the following command:

```bash
python logistic_regression.py
```

or

```bash
py logistic_regression.py
```

---

## Output

The program displays:

- Dataset Information
- Accuracy Score
- Classification Report
- ROC-AUC Score
- Sigmoid Function Explanation
- Threshold Tuning Explanation

The following files are generated automatically:

- confusion_matrix.png
- roc_curve.png

---

## Evaluation Metrics

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## Sigmoid Function

The sigmoid function converts any real-valued number into a probability between 0 and 1.

Formula:

```
σ(x) = 1 / (1 + e^-x)
```

If the probability is greater than or equal to **0.5**, the model predicts **Class 1**; otherwise, it predicts **Class 0**.

---

## Threshold Tuning

The default classification threshold is **0.5**.

- Increasing the threshold generally improves Precision.
- Decreasing the threshold generally improves Recall.

---

## Features

- Binary Classification
- Feature Scaling
- Logistic Regression Model
- Confusion Matrix Visualization
- ROC Curve Visualization
- Exception Handling
- Beginner-Friendly Code
- Clean Code Structure

---

## Future Enhancements

- Add user input for custom predictions.
- Use different classification algorithms for comparison.
- Perform Hyperparameter Tuning.
- Add Cross-Validation.
- Develop a Tkinter GUI for easy interaction.
- Deploy the project as a web application.

---

## Author

**Mopala Nithyasri**

Internship Project – Classification with Logistic Regression