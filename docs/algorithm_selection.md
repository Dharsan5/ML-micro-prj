# Why K-Means Clustering?

## The dataset

The Amazon product dataset contains one row per product with five numeric features extracted after cleaning:

| Feature | Raw column | Cleaning |
|---|---|---|
| Discounted price | `discounted_price` | Strip `₹` and `,`, cast to float |
| Actual price | `actual_price` | Strip `₹` and `,`, cast to float |
| Discount percentage | `discount_percentage` | Strip `%`, cast to float |
| Rating | `rating` | Cast to float |
| Rating count | `rating_count` | Strip `,`, cast to int |

After cleaning, the working dataset has ~1,400 rows and no categorical inputs to the model.

---

## Why not another algorithm?

### DBSCAN
DBSCAN groups points by density and can find arbitrarily shaped clusters without specifying K in advance.
**Why we didn't use it:** Amazon product data clusters by price tier and discount level — these are roughly convex, spherical groupings that match K-Means' assumptions well. DBSCAN also requires careful tuning of `eps` and `min_samples`, which is harder to expose in an interactive dashboard.

### Hierarchical clustering (agglomerative)
Builds a dendrogram of nested merges and can reveal cluster structure at multiple granularities.
**Why we didn't use it:** Scales as O(n²) in memory, making it impractical for datasets above ~1,000 rows without approximate methods. It also does not expose a simple retrain-with-new-K workflow that fits the dashboard's live K selector.

### Gaussian Mixture Models (GMM)
A probabilistic generalisation of K-Means that assigns soft cluster memberships.
**Why we didn't use it:** The extra complexity (covariance matrices, EM convergence) is hard to interpret for a business audience. The goal here is actionable product segments, not probability distributions.

### PCA + K-Means vs. raw K-Means
We apply PCA only for the 2-D *visualisation* of results. The actual clustering is done on all five standardised features, preserving the full signal.

---

## Why K-Means fits this data

### 1. All inputs are numeric and continuous
K-Means computes Euclidean distance between feature vectors. Currency values, percentages, and star ratings are all real numbers — distance is meaningful.

### 2. The clusters are roughly spherical
A scatter of price vs. discount or price vs. rating shows three natural bands: cheap/high-discount, premium/low-discount, and mid-range/popular. These form compact, roughly spherical clouds, which is exactly the geometry K-Means assumes.

### 3. Interpretability matters
Each cluster is summarised by its centroid — a single row of average feature values. A business analyst can immediately read "Cluster 0 has average price ₹350, 62% discount, rating 4.0" and label it *Budget Deals* without any statistical background.

### 4. Scalability
K-Means runs in O(n × k × d × iterations) time. Even rerunning from scratch for every K the user selects in the dashboard takes under a second on this dataset size.

### 5. The elbow + silhouette heuristics are intuitive
WCSS (within-cluster sum of squares) and silhouette score give clear, visual guidance on choosing K. The dashboard exposes both in a side-by-side chart so users can justify their choice of K without any ML background.

---

## Choosing K = 3

The elbow curve shows a clear inflection point at K = 3: inertia drops steeply from K = 2 to K = 3 and then flattens. The silhouette score also peaks (or near-peaks) at K = 3.

Interpretively, three clusters map naturally to a well-known retail segmentation:

| Cluster | Avg price | Avg discount | Avg rating | Label |
|---|---|---|---|---|
| 0 | Low | High (>60%) | Moderate | Budget Deals |
| 1 | High | Low (<30%) | High | Premium Products |
| 2 | Mid | Moderate | High + many reviews | Popular Mid-range |

The dashboard lets users change K from 2 – 9 and auto-refreshes all plots, so the choice of 3 is a recommendation, not a hard constraint.
