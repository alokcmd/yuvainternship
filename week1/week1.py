"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: UCI Adult (Census Income) Dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# 1. DATA ACQUISITION
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

df = pd.concat([X, y], axis=1)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. INITIAL EXPLORATION
print("\nData types:")
print(df.dtypes)

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nDuplicate rows:", df.duplicated().sum())

# Convert whitespace-only strings to missing values.
df = df.replace(r"^\s*$", np.nan, regex=True)

# Standardize string columns by stripping extra whitespace.
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].str.strip()

# 3. MISSING VALUE ANALYSIS
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_pct.round(2)
})

print("\nMissing-value report:")
print(missing_report[missing_report["missing_count"] > 0])

# 4. REMOVE EXACT DUPLICATES
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicates removed: {before - len(df)}")

# 5. HANDLE MISSING VALUES
# For categorical columns, mode is used because categories have no meaningful mean.
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# For numerical columns, median is robust to skew/outliers.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

print("\nRemaining missing values:", int(df.isna().sum().sum()))

# 6. ERRONEOUS / INCONSISTENT ENTRIES
# Normalize the target labels. The test file can contain a trailing period,
# while the training file uses labels without it.
target_col = y.columns[0]
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.rstrip(".")
)

print("\nTarget values after standardization:")
print(df[target_col].value_counts())

# 7. OUTLIER DETECTION USING IQR
# IQR is used because it is simple and robust for skewed numeric variables.
outlier_summary = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_summary.append({
        "column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(count)
    })

outlier_report = pd.DataFrame(outlier_summary)
print("\nOutlier report:")
print(outlier_report)

# Do not automatically delete every statistical outlier.
# Extreme values may be valid observations. Instead, flag them for review.
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[f"{col}_outlier_flag"] = ((df[col] < lower) | (df[col] > upper))

# 8. PREPROCESSING FOR MACHINE LEARNING


target = target_col
X_clean = df.drop(columns=[target])
y_clean = df[target]

numeric_features = X_clean.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_clean.select_dtypes(include="object").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.20, random_state=42, stratify=y_clean
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nTraining shape before preprocessing:", X_train.shape)
print("Training shape after preprocessing:", X_train_processed.shape)
print("Testing shape after preprocessing:", X_test_processed.shape)

# 9. SAVE CLEANED DATA
cleaned_path = "adult_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"\nCleaned dataset saved as: {cleaned_path}")

# 10. OPTIONAL VISUAL CHECKS
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Distribution and Potential Outliers")
plt.tight_layout()
plt.savefig("age_boxplot.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution After Cleaning")
plt.tight_layout()
plt.savefig("age_histogram.png", dpi=200)
plt.show()
