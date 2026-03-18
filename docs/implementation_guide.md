# Implementation Guide

End-to-end walkthrough of the K-Means clustering pipeline, from raw CSV to interactive dashboard.

---

## Step 1 — Load the CSV

```python
import pandas as pd

df = pd.read_csv("Dataset/amazon.csv", encoding="utf-8")
print(df.shape)          # (1465, 16)
print(df.columns.tolist())
# ['product_id', 'product_name', 'category', 'discounted_price',
#  'actual_price', 'discount_percentage', 'rating', 'rating_count', ...]
```

The raw file uses Indian Rupee symbols (`₹`), comma-formatted numbers (`1,099`), and percentage signs (`64%`). These must be stripped before any numeric operation.

---

## Step 2 — Clean numeric columns

```python
def clean_currency(series):
    """Remove ₹ and thousand-separators, return float Series."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace("₹", "", regex=False)
              .str.replace(",", "", regex=False)
              .str.strip(),
        errors="coerce"          # unparseable values become NaN
    )

df["discounted_price"]    = clean_currency(df["discounted_price"])
df["actual_price"]        = clean_currency(df["actual_price"])

df["discount_percentage"] = pd.to_numeric(
    df["discount_percentage"].astype(str)
      .str.replace("%", "", regex=False).str.strip(),
    errors="coerce"
)

df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["rating_count"] = pd.to_numeric(
    df["rating_count"].astype(str)
      .str.replace(",", "", regex=False).str.strip(),
    errors="coerce"
)
```

Extract the top-level product category from the pipe-delimited `category` field:

```python
df["primary_category"] = df["category"].astype(str).str.split("|").str[0]
# "Computers&Accessories|Accessories&Peripherals|..." → "Computers&Accessories"
```

Drop rows where any of the five model features are missing:

```python
FEATURES = ["discounted_price", "actual_price",
            "discount_percentage", "rating", "rating_count"]

df.dropna(subset=FEATURES, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)   # typically (1462, 17)
```

---

## Step 3 — Feature scaling

Raw prices are in the thousands while rating is 1–5. Without scaling, K-Means distance is dominated by price. We apply two transformations:

1. **Log transform** right-skewed columns to compress outliers.
2. **StandardScaler** to give every feature zero mean and unit variance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    X = df[FEATURES].copy()

    # Log-transform skewed columns
    X["discounted_price"] = np.log1p(X["discounted_price"])
    X["actual_price"]     = np.log1p(X["actual_price"])
    X["rating_count"]     = np.log1p(X["rating_count"])

    # Standardise to zero-mean, unit-variance
    return StandardScaler().fit_transform(X)

X_scaled = scale_features(df)
print(X_scaled.shape)    # (1462, 5)
print(X_scaled.mean(axis=0).round(4))  # ≈ [0, 0, 0, 0, 0]
print(X_scaled.std(axis=0).round(4))   # ≈ [1, 1, 1, 1, 1]
```

---

## Step 4 — Find optimal K (elbow + silhouette)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K_RANGE  = range(2, 10)
inertia  = []
sil_scores = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
```

**Inertia (WCSS)** — sum of squared distances from each point to its cluster centroid. Look for the "elbow": the point where adding more clusters returns diminishing gains.

**Silhouette score** — measures how well each point fits its own cluster vs. the nearest competing cluster. Range: –1 (wrong cluster) to +1 (tight, well-separated). Higher is better.

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(list(K_RANGE), inertia, "o-", color="#007AFF")
ax1.axvline(3, color="#FF9500", linestyle="--", label="K=3 chosen")
ax1.set(title="Elbow Curve", xlabel="K", ylabel="Inertia")
ax1.legend()

ax2.plot(list(K_RANGE), sil_scores, "o-", color="#34C759")
ax2.axvline(3, color="#FF9500", linestyle="--", label="K=3 chosen")
ax2.set(title="Silhouette Score", xlabel="K", ylabel="Score")
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/elbow_curve.png", dpi=150, bbox_inches="tight")
```

---

## Step 5 — Fit the final model (K = 3)

```python
K_FINAL = 3

km     = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

df["cluster"] = labels

sil = silhouette_score(X_scaled, labels)
print(f"Silhouette score (K=3): {sil:.4f}")
# → typically 0.20 – 0.30, acceptable for real retail data

# Per-cluster summary in original units
print(df.groupby("cluster")[FEATURES].mean().round(2))
```

`n_init=10` runs the algorithm ten times with different random centroid seeds and keeps the best result (lowest inertia), reducing sensitivity to initialisation.

---

## Step 6 — Visualise clusters with PCA

K-Means operates in 5-D. To plot clusters in 2-D we project with PCA — this is only for visualisation; the cluster labels come from the full 5-D model.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

PALETTE = ["#007AFF", "#FF9500", "#34C759"]

pca = PCA(n_components=2, random_state=42)
X2  = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100

# Project cluster centroids into the same 2-D space
centroids_2d = pca.transform(km.cluster_centers_)

fig, ax = plt.subplots(figsize=(8, 5))

for c in range(K_FINAL):
    mask = labels == c
    ax.scatter(X2[mask, 0], X2[mask, 1],
               color=PALETTE[c], label=f"Cluster {c}",
               alpha=0.45, s=16, edgecolors="none")

ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           s=200, marker="P", c=PALETTE,
           edgecolors="#1C1C1E", linewidths=0.8,
           zorder=5, label="Centroids")

ax.set(xlabel=f"PC1 ({var[0]:.1f}% variance)",
       ylabel=f"PC2 ({var[1]:.1f}% variance)",
       title="K-Means Clusters — PCA 2-D Projection")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/pca_clusters.png", dpi=150, bbox_inches="tight")
```

---

## Step 7 — Interpret the centroids

```python
import seaborn as sns

# Centroid values are already in standardised units
centroid_df = pd.DataFrame(
    km.cluster_centers_,
    columns=["Disc. Price", "Actual Price", "Discount %", "Rating", "Rating Count"],
    index=[f"Cluster {i}" for i in range(K_FINAL)]
)

fig, ax = plt.subplots(figsize=(9, 3))
sns.heatmap(centroid_df, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            cbar_kws={"label": "Standardised value"}, ax=ax)
ax.set_title("Cluster Centroids (positive = above average, negative = below)")
plt.tight_layout()
plt.savefig("outputs/cluster_heatmap.png", dpi=150, bbox_inches="tight")
```

Reading the heatmap:

| Centroid value | Meaning |
|---|---|
| +1.5 | Feature is 1.5 standard deviations above the dataset mean |
| 0.0 | Feature is exactly at the dataset mean |
| –1.2 | Feature is 1.2 standard deviations below the dataset mean |

---

## Step 8 — Export results

```python
# Save clustered DataFrame for downstream use
df.to_csv("outputs/clustered_products.csv", index=False)
print(f"Saved {len(df):,} rows with cluster column to outputs/clustered_products.csv")
```

---

## Full pipeline (condensed)

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load
df = pd.read_csv("Dataset/amazon.csv", encoding="utf-8")

# 2. Clean
def clean_currency(s):
    return pd.to_numeric(
        s.astype(str).str.replace("₹","",regex=False)
                     .str.replace(",","",regex=False).str.strip(),
        errors="coerce")

df["discounted_price"]    = clean_currency(df["discounted_price"])
df["actual_price"]        = clean_currency(df["actual_price"])
df["discount_percentage"] = pd.to_numeric(
    df["discount_percentage"].astype(str).str.replace("%","").str.strip(),
    errors="coerce")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["rating_count"] = pd.to_numeric(
    df["rating_count"].astype(str).str.replace(",","").str.strip(),
    errors="coerce")
df["primary_category"] = df["category"].str.split("|").str[0]

FEATURES = ["discounted_price","actual_price",
            "discount_percentage","rating","rating_count"]
df.dropna(subset=FEATURES, inplace=True)

# 3. Scale
X = df[FEATURES].copy()
X["discounted_price"] = np.log1p(X["discounted_price"])
X["actual_price"]     = np.log1p(X["actual_price"])
X["rating_count"]     = np.log1p(X["rating_count"])
X_scaled = StandardScaler().fit_transform(X)

# 4. Fit
km     = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels

# 5. Evaluate
print(f"Silhouette: {silhouette_score(X_scaled, labels):.4f}")
print(df.groupby("cluster")[FEATURES].mean().round(2))

# 6. Export
df.to_csv("outputs/clustered_products.csv", index=False)
```

---

## File structure

```
ML-Micro-project/
├── Dataset/
│   └── amazon.csv              Raw input data
├── docs/
│   ├── README.md               This index
│   ├── algorithm_selection.md  Why K-Means was chosen
│   └── implementation_guide.md This file
├── outputs/
│   ├── elbow_curve.png
│   ├── pca_clusters.png
│   ├── cluster_heatmap.png
│   └── clustered_products.csv
└── dashboard.py                Interactive Tkinter dashboard
```
