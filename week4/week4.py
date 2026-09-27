# ============================================================
# WEEK 4: SUPERVISED LEARNING MODEL IMPLEMENTATION
# Project: Breast Cancer Classification using Logistic Regression
# ============================================================

# ------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(X.head())

print("\nDataset shape:")
print(X.shape)

print("\nTarget names:")
print(data.target_names)


# ------------------------------------------------------------
# 3. Basic Dataset Information
# ------------------------------------------------------------

print("\nDataset Information:")
print(X.info())

print("\nStatistical Summary:")
print(X.describe())

print("\nMissing Values:")
print(X.isnull().sum().sum())


# ------------------------------------------------------------
# 4. Target/Class Distribution
# ------------------------------------------------------------

print("\nClass Distribution:")
print(y.value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(x=y)

plt.title("Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Samples")

plt.show()


# ------------------------------------------------------------
# 5. Feature Visualization
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=X.iloc[:, :10]
)

plt.title("Distribution of First 10 Features")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 6. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# ------------------------------------------------------------
# 7. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ------------------------------------------------------------
# 8. Create Logistic Regression Model
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)


# ------------------------------------------------------------
# 9. Train Model
# ------------------------------------------------------------

model.fit(
    X_train_scaled,
    y_train
)

print("\nModel training completed.")


# ------------------------------------------------------------
# 10. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\nPredictions:")
print(y_pred[:20])


# ------------------------------------------------------------
# 11. Calculate Evaluation Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ------------------------------------------------------------
# 12. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ------------------------------------------------------------
# 13. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.show()


# ------------------------------------------------------------
# 14. ROC Curve
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 15. Cross-Validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n====================================")
print("5-FOLD CROSS-VALIDATION")
print("====================================")

print("Cross-validation scores:")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(
    f"\nMean CV Accuracy: {cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: {cv_scores.std():.4f}"
)


# ------------------------------------------------------------
# 16. Feature Importance using Model Coefficients
# ------------------------------------------------------------

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
})

feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10)
)


# ------------------------------------------------------------
# 17. Visualize Top Features
# ------------------------------------------------------------

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Absolute_Coefficient",
    y="Feature"
)

plt.title(
    "Top 10 Features Based on Logistic Regression Coefficients"
)

plt.xlabel("Absolute Coefficient")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. Actual vs Predicted Results
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(20))


# ------------------------------------------------------------
# 19. Final Results Summary
# ------------------------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "Mean CV Accuracy"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        cv_scores.mean()
    ]
})

print("\n====================================")
print("FINAL RESULTS")
print("====================================")

print(results)


# ------------------------------------------------------------
# 20. Save Results to CSV
# ------------------------------------------------------------

results.to_csv(
    "week4_model_results.csv",
    index=False
)

print(
    "\nResults saved as: week4_model_results.csv"
)


# ------------------------------------------------------------
# 21. Final Message
# ------------------------------------------------------------

print("\n====================================")
print("WEEK 4 PROJECT COMPLETED")
print("====================================")
print("Dataset       : Breast Cancer Wisconsin")
print("Algorithm     : Logistic Regression")
print("Validation    : 5-Fold Cross-Validation")
print("Evaluation    : Accuracy, Precision, Recall, F1, ROC-AUC")
print("====================================")