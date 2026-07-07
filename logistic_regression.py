"""
Project: Classification with Logistic Regression
Author: Mopala Nithyasri
Python Version: 3.13.1
"""

# ==========================
# Import Required Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==========================
# Main Program
# ==========================
try:
    # Load the Breast Cancer dataset
    dataset = load_breast_cancer()

    # Create a DataFrame
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # Add target column
    df["Target"] = dataset.target

    # Display first 5 rows
    print("\nFirst Five Rows\n")
    print(df.head())

    # Display dataset shape
    print("\nDataset Shape:", df.shape)

    # Check missing values
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Separate features and target
    X = df.drop("Target", axis=1)
    y = df["Target"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)

    # Predict classes
    y_pred = model.predict(X_test)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy Score")
    print(accuracy)

    # Classification Report
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.target_names
    )

    disp.plot()

    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    plt.show()

    # ROC Curve
    fpr, tpr, threshold = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label="ROC Curve")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("roc_curve.png")

    plt.show()

    # ROC AUC Score
    auc_score = roc_auc_score(y_test, y_prob)

    print("\nROC AUC Score")
    print(auc_score)

    # Sigmoid Function Explanation
    print("\nSigmoid Function")
    print("-------------------------")
    print("The sigmoid function converts values into probabilities between 0 and 1.")
    print("Formula: σ(x) = 1 / (1 + e^-x)")
    print("If probability >= 0.5 → Class 1")
    print("If probability < 0.5 → Class 0")

    # Threshold Explanation
    print("\nThreshold Tuning")
    print("-------------------------")
    print("Default Threshold = 0.5")
    print("Increasing threshold improves precision.")
    print("Decreasing threshold improves recall.")

    print("\nProject Completed Successfully!")

except FileNotFoundError:
    print("Dataset file not found.")

except PermissionError:
    print("Permission denied while saving output files.")

except Exception as error:
    print("\nAn unexpected error occurred.")
    print(error)