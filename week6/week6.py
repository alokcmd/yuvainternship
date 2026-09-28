"""
===============================================================================
INTEGRATIVE CAPSTONE PROJECT: END-TO-END DATA SCIENCE PIPELINE
Domain: Telecommunications Customer Retention & Churn Analytics
Author: Data Science Intern
Estimated Hours: 30 - 35 Hours
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

# Set global visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

# =============================================================================
# PHASE 1 & 2: DATA ACQUISITION, PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

def load_and_preprocess_data():
    """
    Simulates loading IBM Telco Customer Churn data, performs schema validation,
    handles missing values, and constructs engineered features.
    """
    np.random.seed(42)
    n_samples = 1000

    # Dataset Schema Construction
    customer_ids = [f"CUST-{1000+i}" for i in range(n_samples)]
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.25, 118.75, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.uniform(-50, 50, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], size=n_samples, p=[0.45, 0.35, 0.20])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples)
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.3, 0.5, 0.2])

    # Probability-driven Target Variable Generation (Realistic Churn Bias)
    churn_prob = (0.15 + 0.35 * (contract_types == 'Month-to-month') 
                  + 0.25 * (internet_service == 'Fiber optic') 
                  - 0.20 * (tenure > 24) 
                  - 0.15 * (tech_support == 'Yes'))
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_types,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'TechSupport': tech_support,
        'Churn': churn
    })

    # Feature Engineering
    df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 Yr', '1-2 Yrs', '2-4 Yrs', '4+ Yrs'])
    df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)
    
    return df

# =============================================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df):
    """
    Executes Exploratory Data Analysis and generates diagnostic visual charts.
    """
    print(f"Dataset Overview:\n{df.info()}\n")
    print(f"Summary Statistics:\n{df.describe()}\n")
    print(f"Target Distribution:\n{df['Churn'].value_counts(normalize=True)}\n")

    # Chart 1: Churn Breakdown & Tenure Density
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x='Churn', palette=['#1f77b4', '#d62728'], ax=axes[0])
    axes[0].set_title('Overall Customer Churn Distribution', fontweight='bold')
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])

    sns.kdeplot(data=df, x='Tenure', hue='Churn', palette=['#1f77b4', '#d62728'], common_norm=False, fill=True, ax=axes[1])
    axes[1].set_title('Tenure Density Curve by Churn Status', fontweight='bold')
    plt.tight_layout()
    plt.show()

# =============================================================================
# PHASE 4: UNSUPERVISED CLUSTERING (K-MEANS & PCA)
# =============================================================================

def perform_unsupervised_clustering(X_num):
    """
    Performs PCA dimensionality reduction and K-Means behavioral clustering.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title('Customer Behavioral Segments (PCA Projection)', fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.show()

    return clusters

# =============================================================================
# PHASE 5 & 6: SUPERVISED MODELING & EVALUATION
# =============================================================================

def train_and_evaluate_models(df):
    """
    Builds data preprocessing pipeline and evaluates Logistic Regression,
    Random Forest, and XGBoost models.
    """
    X = df.drop(columns=['CustomerID', 'Churn', 'Tenure_Group'])
    y = df['Churn']

    num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyToTotalRatio']
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']

    # Unsupervised Clustering Execution
    perform_unsupervised_clustering(X[num_cols])

    # Preprocessing Pipeline
    num_transformer = Pipeline([('scaler', StandardScaler())])
    cat_transformer = Pipeline([('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Classifiers Configuration
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    }

    results = []

    for name, clf in classifiers.items():
        clf.fit(X_train_proc, y_train)
        preds = clf.predict(X_test_proc)
        probs = clf.predict_proba(X_test_proc)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    results_df = pd.DataFrame(results)
    print("\n================ MODEL PERFORMANCE COMPARISON ================")
    print(results_df.to_string(index=False))

# Run Full Pipeline
if __name__ == "__main__":
    df_data = load_and_preprocess_data()
    perform_eda(df_data)
    train_and_evaluate_models(df_data)