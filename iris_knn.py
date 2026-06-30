"""
==============================================
Artificial Intelligence Project 2
Title : Iris Flower Classification Using KNN
Author : Abhinav Soni
Internship : DecodeLabs AI Internship
==============================================
"""

# -------------------------------
# Import Libraries
# -------------------------------

import numpy as np
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

# -------------------------------
# Load Dataset
# -------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("="*60)
print("IRIS DATASET INFORMATION")
print("="*60)

print("\nFeatures:")
print(feature_names)

print("\nClasses:")
print(class_names)

print("\nTotal Samples:", len(X))

# -------------------------------
# Convert to DataFrame
# -------------------------------

df = pd.DataFrame(X, columns=feature_names)
df["Species"] = y

print("\nFirst Five Records\n")
print(df.head())

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True

)

# -------------------------------
# Build KNN Model
# -------------------------------

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Accuracy
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : {:.2f}%".format(accuracy*100))

# -------------------------------
# F1 Score
# -------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("F1 Score :", round(f1,4))

# -------------------------------
# Classification Report
# -------------------------------

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

# -------------------------------
# Plot Confusion Matrix
# -------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("output.png")

plt.show()

# -------------------------------
# Predict New Flower
# -------------------------------

sample = [[5.1,3.5,1.4,0.2]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nPrediction For New Flower :")

print(class_names[prediction[0]])

print("\nProject Completed Successfully.")