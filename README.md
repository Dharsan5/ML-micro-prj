# Amazon Product K-Means Clustering Dashboard

An interactive desktop dashboard for exploring K-Means clustering on the Amazon Product dataset. Built with Python, Tkinter, Matplotlib, and scikit-learn — distributed as a standalone Windows EXE.

---

## Quick Start

1. Run `dist/AmazonClusteringDashboard.exe`
2. The dashboard loads `Dataset/amazon.csv` automatically
3. Use the sidebar to explore views and adjust K

No Python installation required to run the EXE.

---

## Project Structure

```
ML-Micro-project/
├── Dataset/
│   └── amazon.csv                   # Source dataset (Amazon products)
├── dist/
│   └── AmazonClusteringDashboard.exe  # Standalone Windows executable
├── outputs/                         # Saved plots (PNG/SVG/PDF)
├── dashboard.py                     # Full application source
└── README.md
```

---

## Dataset

**File:** `Dataset/amazon.csv`
**Source:** Amazon Product Reviews & Pricing Dataset
**Rows:** ~1,500 products after cleaning

| Column | Raw Format | Cleaned To |
|---|---|---|
| `discounted_price` | `₹1,299` | `float` |
| `actual_price` | `₹1,999` | `float` |
| `discount_percentage` | `35%` | `float` |
| `rating` | `4.2` | `float` |
| `rating_count` | `24,269` | `int` |
| `category` | `Electronics\|Cables\|...` | primary category extracted |

---

## How It Works

### Data Pipeline

```
Raw CSV → Clean (strip ₹, %, ,) → Drop NaN rows
       → Log-transform skewed columns (price, rating_count)
       → StandardScaler → X_scaled (5 features)
```

Prices and rating counts are log-transformed before scaling to reduce the influence of extreme outliers on cluster assignments.

### Clustering

- Algorithm: **K-Means** (`sklearn.cluster.KMeans`)
- Default K: **3** (adjustable via slider, range 2–8)
- Init: `n_init=10`, `random_state=42` for reproducibility
- Optimal K chosen via **Elbow curve** (WCSS) + **Silhouette score**

### Dimensionality Reduction

PCA (2 components) is applied only for the scatter plot visualisation — clustering itself runs on all 5 scaled features.

---

## Dashboard Controls

| Control | Description |
|---|---|
| **K Slider** | Set number of clusters (2–8) |
| **Run Clustering** | Re-fit K-Means with current K and refresh view |
| **Browse CSV** | Load a different CSV file |
| **Save Plot** | Export the current plot (PNG / SVG / PDF) |
| **Matplotlib toolbar** | Zoom, pan, reset on every plot |

---

## Available Views

| View | What it shows |
|---|---|
| Elbow + Silhouette | WCSS and silhouette score for K = 2–10, used to justify K choice |
| PCA Scatter | 2-D projection of all products coloured by cluster, with centroids marked |
| Centroid Heatmap | Standardised feature values at each cluster centre |
| Cluster Profiles | Normalised bar chart comparing feature means across clusters |
| Rating vs Discount | Scatter coloured by cluster — reveals discount/quality trade-offs |
| Price vs Discount | Scatter coloured by cluster — shows pricing strategy segments |
| Correlation Heatmap | Pairwise Pearson correlations between all 5 features |
| Price Distribution | Histogram of discounted price (97th-percentile clipped) |
| Discount Distribution | Histogram of discount percentage |
| Top Categories | Horizontal bar — average rating per category (min 5 products) |
| Segment Pie | Proportion of products in each cluster |

---

## Cluster Interpretation (K = 3)

Cluster labels are assigned automatically based on relative price rank:

| Segment | Typical Profile |
|---|---|
| **Budget Deals** | Low discounted price, high discount %, moderate rating |
| **Popular Mid-range** | Mid price, moderate discount, high review count |
| **Premium Products** | High actual price, low discount %, high rating |

These labels update in the sidebar stats panel after every clustering run.

---

## Rebuilding the EXE

Requirements: Python 3.10+

```bash
pip install pandas numpy matplotlib scikit-learn seaborn pyinstaller

pyinstaller --onefile --windowed \
  --name "AmazonClusteringDashboard" \
  --add-data "Dataset/amazon.csv;Dataset" \
  --exclude-module sqlalchemy \
  --exclude-module IPython \
  dashboard.py
```

Output: `dist/AmazonClusteringDashboard.exe`

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn (KMeans, StandardScaler, PCA) |
| Visualisation | matplotlib, seaborn |
| GUI | tkinter (built-in) |
| Packaging | PyInstaller |
