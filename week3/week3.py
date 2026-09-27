# ============================================================
# WEEK 3 TASK: UNSUPERVISED LEARNING AND CLUSTERING ANALYSIS
# Algorithm: K-Means Clustering
# Dataset: Iris Dataset
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


# -----------------------------
# 2. Load Iris Dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# -----------------------------
# 3. Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 4. Feature Distribution
# -----------------------------
plt.figure(figsize=(10, 7))

df.hist(figsize=(10, 7), bins=15)

plt.suptitle("Distribution of Iris Dataset Features")
plt.tight_layout()
plt.show()


# -----------------------------
# 5. Data Standardization
# -----------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data:")
print(X_scaled[:5])


# -----------------------------
# 6. Find Optimal Number of Clusters
#    Using Elbow Method
# -----------------------------

inertia = []

K = range(2, 8)

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia__)


# Plot Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(K, inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.grid(True)
plt.show()


# -----------------------------
# 7. Silhouette Score
# -----------------------------

silhouette_scores = []

for k in K:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


# Plot Silhouette Scores
plt.figure(figsize=(8, 5))

plt.plot(
    K,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.title("Silhouette Score for Different K Values")

plt.grid(True)
plt.show()


# -----------------------------
# 8. Apply K-Means
# -----------------------------

optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# Add cluster labels to dataframe
df["Cluster"] = clusters

print("\nCluster Labels:")
print(df["Cluster"].value_counts().sort_index())


# -----------------------------
# 9. Final Silhouette Score
# -----------------------------

final_silhouette = silhouette_score(
    X_scaled,
    clusters
)

print("\nFinal Silhouette Score:")
print(round(final_silhouette, 4))


# -----------------------------
# 10. PCA for Visualization
# -----------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PCA1", "PCA2"]
)

pca_df["Cluster"] = clusters


# -----------------------------
# 11. Visualize Clusters
# -----------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=pca_df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("K-Means Clustering of Iris Dataset")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(title="Cluster")

plt.grid(True)
plt.show()


# -----------------------------
# 12. Cluster Characteristics
# -----------------------------

cluster_means = df.groupby("Cluster").mean()

print("\nCluster Characteristics:")
print(cluster_means)


# -----------------------------
# 13. Cluster Mean Visualization
# -----------------------------

cluster_means.T.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Average Feature Values by Cluster")

plt.xlabel("Features")
plt.ylabel("Average Value")

plt.xticks(rotation=45)

plt.legend(title="Cluster")

plt.tight_layout()
plt.show()


# -----------------------------
# 14. Compare Clusters
#    With Original Iris Labels
# -----------------------------

comparison = pd.crosstab(
    pd.Series(y, name="Actual Species"),
    pd.Series(clusters, name="Cluster")
)

print("\nActual Species vs Cluster:")
print(comparison)


# -----------------------------
# 15. Heatmap
# -----------------------------

plt.figure(figsize=(8, 5))

sns.heatmap(
    comparison,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Actual Species vs K-Means Clusters")

plt.tight_layout()
plt.show()


# -----------------------------
# 16. Adjusted Rand Index
# -----------------------------

ari = adjusted_rand_score(y, clusters)

print("\nAdjusted Rand Index:")
print(round(ari, 4))


# -----------------------------
# 17. Hierarchical Clustering
# -----------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print("\nHierarchical Clustering Silhouette Score:")
print(round(hierarchical_score, 4))


# -----------------------------
# 18. Hierarchical Clustering Plot
# -----------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=hierarchical_labels,
    palette="Set2",
    s=100
)

plt.title("Hierarchical Clustering of Iris Dataset")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)
plt.show()


# -----------------------------
# 19. Final Results
# -----------------------------

print("\n====================================")
print("FINAL CLUSTERING RESULTS")
print("====================================")

print("Number of Clusters:", optimal_k)

print(
    "K-Means Silhouette Score:",
    round(final_silhouette, 4)
)

print(
    "Adjusted Rand Index:",
    round(ari, 4)
)

print(
    "Hierarchical Silhouette Score:",
    round(hierarchical_score, 4)
)

print("\nCluster Sizes:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster Means:")

print(cluster_means)

print("\nAnalysis Completed Successfully!")