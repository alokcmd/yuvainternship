# ============================================================
# WEEK 2 TASK: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Wine Recognition Dataset
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

# ============================================================
# 2. Load Dataset
# ============================================================

wine = load_wine(as_frame=True)

df = wine.frame

# Convert target numbers into readable class names
df["target_name"] = df["target"].map(
    dict(enumerate(wine.target_names))
)

print("Dataset loaded successfully!")
# 3. Display First 5 Rows


print("\nFirst 5 rows:")
print(df.head())

# 4. Dataset Shape

print("\nDataset Shape:")
print(df.shape)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ============================================================
# 5. Dataset Information
# ============================================================

print("\nDataset Information:")
print(df.info())


# ============================================================
# 6. Check Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 7. Statistical Summary
# ============================================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 8. Column Names
# ============================================================

print("\nColumn Names:")
print(df.columns)


# ============================================================
# 9. Target / Wine Class Distribution
# ============================================================

print("\nWine Class Distribution:")
print(df["target_name"].value_counts())


# ============================================================
# 10. Visualization 1
# Wine Class Distribution
# ============================================================

plt.figure(figsize=(8, 5))

class_counts = df["target_name"].value_counts()

class_counts.plot(kind="bar")

plt.title("Distribution of Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 11. Visualization 2
# Alcohol Content by Wine Class
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="target_name",
    y="alcohol"
)

plt.title("Alcohol Content by Wine Class")
plt.xlabel("Wine Class")
plt.ylabel("Alcohol")

plt.tight_layout()
plt.show()


# ============================================================
# 12. Visualization 3
# Flavanoids vs Color Intensity
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="flavanoids",
    y="color_intensity",
    hue="target_name"
)

plt.title("Flavanoids vs Color Intensity")
plt.xlabel("Flavanoids")
plt.ylabel("Color Intensity")
plt.legend(title="Wine Class")

plt.tight_layout()
plt.show()


# ============================================================
# 13. Visualization 4
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(12, 8))

# Select only numerical features
numeric_data = df[wine.feature_names]

correlation = numeric_data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Wine Features")

plt.tight_layout()
plt.show()


# ============================================================
# 14. Visualization 5
# Average Features by Wine Class
# ============================================================

selected_features = [
    "alcohol",
    "malic_acid",
    "flavanoids",
    "color_intensity"
]

mean_values = df.groupby(
    "target_name"
)[selected_features].mean()

print("\nAverage Feature Values:")
print(mean_values)


mean_values.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Average Chemical Features by Wine Class")
plt.xlabel("Wine Class")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.legend(title="Features")

plt.tight_layout()
plt.show()


# ============================================================
# 15. Feature Distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="alcohol",
    kde=True
)

plt.title("Distribution of Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# 16. Pairplot
# ============================================================

sns.pairplot(
    df,
    vars=[
        "alcohol",
        "malic_acid",
        "flavanoids",
        "color_intensity"
    ],
    hue="target_name"
)

plt.show()


# ============================================================
# 17. Find Highest Average Alcohol Class
# ============================================================

alcohol_average = df.groupby(
    "target_name"
)["alcohol"].mean()

print("\nAverage Alcohol by Class:")
print(alcohol_average)

highest_alcohol_class = alcohol_average.idxmax()

print(
    "\nClass with highest average alcohol:",
    highest_alcohol_class
)


# ============================================================
# 18. Find Highest Average Flavanoids Class
# ============================================================

flav_average = df.groupby(
    "target_name"
)["flavanoids"].mean()

print("\nAverage Flavanoids by Class:")
print(flav_average)

highest_flav_class = flav_average.idxmax()

print(
    "\nClass with highest average flavanoids:",
    highest_flav_class
)


# ============================================================
# 19. Find Strong Correlations
# ============================================================

print("\nCorrelation Matrix:")
print(correlation)


# ============================================================
# 20. Final Summary
# ============================================================

print("\n====================================")
print("EDA ANALYSIS COMPLETED")
print("====================================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())
print("Number of Classes:", df["target_name"].nunique())

print("\nWine Classes:")
print(df["target_name"].unique())

print("\nAnalysis completed successfully!")