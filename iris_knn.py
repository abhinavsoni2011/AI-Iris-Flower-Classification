"""
============================================================
Project 2 : Iris Flower Classification using K-Nearest Neighbors (KNN)

Author      : Abhinav Soni
Internship  : DecodeLabs AI Internship 2026

Description:
This project classifies Iris flower species using the
K-Nearest Neighbors (KNN) Machine Learning algorithm.
============================================================
"""

# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)


def main():

    # ---------------------------------------------------------
    # Load Iris Dataset
    # ---------------------------------------------------------

    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    class_names = iris.target_names

    print("=" * 60)
    print("IRIS FLOWER CLASSIFICATION USING KNN")
    print("=" * 60)

    print("\nDataset Information")
    print("-" * 60)

    print(f"Total Samples : {len(X)}")
    print(f"Features      : {len(feature_names)}")
    print(f"Classes       : {len(class_names)}")

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(X, columns=feature_names)
    df["Species"] = y

    print("\nFirst Five Records\n")
    print(df.head())

    # ---------------------------------------------------------
    # Feature Scaling
    # ---------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # ---------------------------------------------------------
    # Train KNN Model
    # ---------------------------------------------------------

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy * 100:.2f}%")

    # ---------------------------------------------------------
    # F1 Score
    # ---------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(f"F1 Score : {f1:.4f}")

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names
        )
    )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix\n")
    print(cm)

    # ---------------------------------------------------------
    # Plot Confusion Matrix
    # ---------------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig("output.png")

    plt.show()

    # ---------------------------------------------------------
    # Test with New Sample
    # ---------------------------------------------------------

    sample = [[5.1, 3.5, 1.4, 0.2]]

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    print("\n" + "=" * 60)
    print("NEW SAMPLE PREDICTION")
    print("=" * 60)

    print(f"Predicted Species : {class_names[prediction[0]]}")

    # ---------------------------------------------------------
    # End
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("Project Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
