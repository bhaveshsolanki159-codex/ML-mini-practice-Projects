# ⌚ Smartwatch Discount Price Prediction & ML Pipeline (Improved Version)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-red.svg?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

An end-to-end Machine Learning project that predicts smartwatch discount prices and analyzes pricing dynamics based on brand prestige, physical specifications (weight, display dimensions, dial shape, strap material), connectivity features, and customer engagement metrics.

---

## 📌 Table of Contents

- [Overview & Objectives](#-overview--objectives)
- [System Architecture & ML Lifecycle](#-system-architecture--ml-lifecycle)
- [Repository Structure](#-repository-structure)
- [Datasets & Feature Transformation](#-datasets--feature-transformation)
- [Exploratory Data Analysis (EDA) Highlights](#-exploratory-data-analysis-eda-highlights)
- [Feature Engineering & Data Preprocessing](#-feature-engineering--data-preprocessing)
- [Model Evaluation & Benchmarking](#-model-evaluation--benchmarking)
- [Hyperparameter Optimization](#-hyperparameter-optimization)
- [Installation & Getting Started](#-installation--getting-started)
- [Model Inference Quickstart](#-model-inference-quickstart)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 Overview & Objectives

E-commerce pricing strategies for smart devices depend on a complex interaction of hardware capabilities, consumer ratings, brand positioning, and market segment. This project provides a robust, reproducible regression pipeline designed to:
1. **Analyze Pricing Determinants**: Identify how physical specs (display size, weight), battery life, brand tier, and customer ratings drive discount pricing.
2. **Handle Real-World Messy Data**: Parse unstructured string values (e.g. ranges like `20 - 35 g`, units like `1.8 inches`), eliminate multivariate outliers via IQR, and impute missing specs hierarchically using brand-level statistics.
3. **Train & Tune Multiple Regression Models**: Benchmark Linear Regression, Decision Trees, Random Forest Regressors, and XGBoost using $K$-Fold cross-validation.
4. **Deploy a Production-Grade Pipeline**: Encapsulate feature transformers and the tuned model into a unified Scikit-Learn `Pipeline` serialized to `model.pkl`.

---

## 🏗️ System Architecture & ML Lifecycle

```mermaid
flowchart TD
    A[Raw Dataset: smartwatches.csv<br/>450 rows × 16 cols] --> B[Phase 1: Exploratory Data Analysis<br/>main.ipynb]
    B -->|Distributions, Skewness, Correlation| C[Phase 2: Data Cleaning & Feature Engineering<br/>feature_engineering.ipynb]
    
    C --> C1[Regex String Extraction: Display Size & Weight]
    C --> C2[Target Derivation: Discount Price]
    C --> C3[IQR Outlier Filtering: 450 → 181 rows]
    C --> C4[Hierarchical Imputation: Brand Median + Global Median]
    C --> C5[MinMax Normalization & OneHotEncoding]
    
    C5 --> D[Engineered Dataset: cleaned.csv<br/>181 rows × 204 cols]
    
    D --> E[Phase 3: Model Selection & Evaluation<br/>modeling.ipynb]
    E --> E1[Linear Regression Baseline]
    E --> E2[Decision Tree Regressor]
    E --> E3[Random Forest Regressor]
    E --> E4[XGBoost Regressor]
    
    E4 --> F[5-Fold Cross-Validation & GridSearchCV Tuning]
    F -->|Best Params: lr=0.1, depth=3, n_est=300| G[Tuned Pipeline<br/>CV R² = 0.8164 | Test R² = 0.7588]
    G --> H[Serialized Production Model: model.pkl]
```

---

## 📂 Repository Structure

```text
Watch Price Predictions Improved Version/
├── Datasets/
│   ├── README.md               # Detailed dataset documentation & feature dictionary
│   ├── smartwatches.csv        # Raw scraped e-commerce smartwatch dataset (450 rows × 16 cols)
│   └── cleaned.csv             # Processed, outlier-filtered & encoded dataset (181 rows × 204 cols)
├── feature_engineering.ipynb   # Regex extraction, IQR outlier removal, imputation & encoding
├── main.ipynb                  # EDA, profiling, distribution analysis & correlation matrix
├── modeling.ipynb              # Baseline models, 5-fold CV, GridSearchCV tuning & export
├── model.pkl                   # Exported scikit-learn Pipeline (Preprocessor + Tuned XGBoost)
├── requirements.txt            # Python dependencies and package versions
└── README.md                   # Main project documentation
```

---

## 📊 Datasets & Feature Transformation

| Feature Name | Raw Data Type | Transformed / Engineered Representation | Description |
| :--- | :--- | :--- | :--- |
| `Brand` | Categorical (String) | One-Hot Encoded | Manufacturer / Brand (Noise, Fire-Boltt, boAt, Apple, etc.) |
| `Current Price` | Numeric (`float64`) | MinMax Scaled ($[0, 1]$) | Selling price listed on the marketplace (₹) |
| `Original Price` | Numeric (`float64`) | MinMax Scaled ($[0, 1]$) | Manufacturer Suggested Retail Price (MSRP / MRP) |
| `Discount Percentage` | Numeric (`float64`) | Used to compute Target | Discount percentage listed on product page |
| `Rating` | Numeric (`float64`) | MinMax Scaled ($[0, 1]$) | Average user rating out of 5.0 |
| `Number OF Ratings` | Numeric (`float64`) | MinMax Scaled ($[0, 1]$) | Total volume of consumer reviews |
| `Display Size` | String (`"1.4 inches"`) | Numeric `Display Size (Inches)` | Screen diagonal extracted via regex |
| `Weight` | String (`"20 - 35 g"`) | Numeric `Weight (g)` | Weight range mapped to midpoint or bound values |
| `Battery Life (Days)` | Numeric (`float64`) | MinMax Scaled ($[0, 1]$) | Battery endurance on single charge |
| `Dial Shape` | Categorical (String) | One-Hot Encoded | Physical shape (Circle, Square, Rectangle, etc.) |
| `Strap Material` | Categorical (String) | One-Hot Encoded | Material (Silicon, Stainless Steel, Leather, etc.) |
| `Touchscreen` | Categorical (`Yes`/`No`) | One-Hot Encoded | Presence of touch display |
| `Bluetooth` | Categorical (`Yes`/`No`) | One-Hot Encoded | Bluetooth calling / connectivity support |
| **`Discount Price`** | *Engineered Target* | Continuous numeric (₹) | **Target Variable**: `Original Price * (-Discount Percentage) / 100` |

> For in-depth data dictionaries and transformation lineage, refer to [Datasets/README.md](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/Datasets/README.md).

---

## 🔍 Exploratory Data Analysis (EDA) Highlights

Exploratory analysis conducted in [`main.ipynb`](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/main.ipynb):
- **Distribution & Skewness**: Evaluated kernel density estimates (KDE) and skewness across all continuous attributes. Strong positive skewness was observed in `Number OF Ratings` and `Current Price`.
- **Multicollinearity & Correlations**: Identified high correlation between `Current Price`, `Original Price`, and the calculated `Discount Price`.
- **Automated Profiling**: Leveraged `ydata-profiling` (`ProfileReport`) for missing-value patterns, high-cardinality flags, and interaction plots.
- **Categorical Influence (ANOVA)**: Computed one-way ANOVA $F$-tests to statistically confirm the relationship between categorical features (`Brand`, `Dial Shape`, `Strap Material`) and discount margins ($p < 0.05$).

---

## 🛠️ Feature Engineering & Data Preprocessing

Implemented systematically in [`feature_engineering.ipynb`](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/feature_engineering.ipynb):

1. **Unstructured String Parsing**:
   - Extracted float values from strings like `"1.8 inches"` $\rightarrow `1.8`$.
   - Converted weight categories (`"20 - 35 g"` $\rightarrow `27.5`$, `"50 - 75 g"` $\rightarrow `62.5`$, `"75g +"` $\rightarrow `75.0`$, `"<= 20 g"` $\rightarrow `20.0`$).
2. **IQR Outlier Removal**:
   - Filtered multivariate outliers on `['Current Price', 'Original Price', 'Rating', 'Number OF Ratings', 'Display Size']` using $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
   - Refined sample count from 450 to 181 high-density records.
3. **Hierarchical Brand-Aware Imputation**:
   - Missing technical specs (`Display Size`, `Weight`, `Battery Life`, `Rating`) were imputed with the **Brand Median**, falling back to the **Global Median** if brand records were insufficient.
   - Categorical missing values (`Dial Shape`, `Strap Material`, `Touchscreen`) were imputed with `'Unknown'`.
4. **Feature Scaling & One-Hot Encoding**:
   - Scaled numerical attributes to $[0, 1]$ using Scikit-Learn's `MinMaxScaler`.
   - Converted categorical attributes into binary dummy vectors using `OneHotEncoder(handle_unknown='ignore')`.

---

## 📈 Model Evaluation & Benchmarking

Four regression architectures were evaluated on an 80/20 train-test split and validated using 5-Fold Cross-Validation in [`modeling.ipynb`](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/modeling.ipynb):

### 1. Test Split Performance Benchmark

| Model | MAE (₹) | RMSE (₹) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Tuned XGBoost Regressor** | **684.21** | **1,412.50** | **0.7588** |
| XGBoost (Default) | 964.53 | 1,894.74 | 0.6931 |
| Decision Tree Regressor | 1,287.81 | 2,252.05 | 0.5665 |
| Random Forest Regressor | 1,218.83 | 2,352.03 | 0.5272 |
| Linear Regression *(Baseline)* | ~0.00 | ~0.00 | 1.0000* |

*\*Note: Linear Regression achieves a near-perfect score due to the direct algebraic identity between Original Price and Current Price in calculating Discount Price.*

### 2. 5-Fold Cross-Validation Comparison

| Model | 5-Fold CV Mean $R^2$ | CV Standard Deviation ($\sigma$) |
| :--- | :---: | :---: |
| **XGBoost (Tuned with GridSearchCV)** | **0.8164** | **± 0.0421** |
| XGBoost (Default) | 0.7278 | ± 0.0649 |
| Random Forest Regressor | 0.6233 | ± 0.1320 |
| Decision Tree Regressor | 0.5778 | ± 0.1174 |
| Linear Regression | 0.9241 | ± 0.0778 |

---

## ⚙️ Hyperparameter Optimization

A exhaustive 5-fold cross-validated grid search (`GridSearchCV`) was performed over 27 parameter candidates (135 total fits):

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2]
}
```

### Optimal Configuration Found:
- **`learning_rate`**: `0.1`
- **`max_depth`**: `3` (prevents tree overfitting on smaller tabular feature sets)
- **`n_estimators`**: `300`
- **Best Cross-Validation Score ($R^2$)**: **`0.8164` (81.64%)**

The optimized estimator was exported as a production pipeline to [`model.pkl`](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/model.pkl).

---

## 🚀 Installation & Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/bhaveshsolanki159-codex/ML-mini-practice-Projects.git
cd "Watch Price Predictions Improved Version"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Running the Jupyter Notebooks

Launch JupyterLab or Jupyter Notebook to execute the pipeline stages interactively:

```bash
jupyter lab
```

Execute the notebooks in sequence:
1. `main.ipynb` (Exploratory Data Analysis & Statistics)
2. `feature_engineering.ipynb` (Cleaning, Imputation & Feature Extraction)
3. `modeling.ipynb` (Model Training, Tuning & Serialization)

---

## 💻 Model Inference Quickstart

You can load and make predictions using the pre-trained `model.pkl` in just a few lines of Python:

```python
import pickle
import pandas as pd

# 1. Load the serialized pipeline
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# 2. Load test or new sample data
df = pd.read_csv('Datasets/cleaned.csv')
X = df.drop(columns=['Discount Price'])

# 3. Predict discount pricing
sample = X.iloc[:5]
predicted_discount = pipeline.predict(sample)

for i, pred in enumerate(predicted_discount):
    print(f"Sample {i+1} Estimated Discount Price: ₹{abs(pred):,.2f}")
```

---

## 🔮 Future Roadmap

- [ ] **Streamlit / FastAPI Web App**: Develop an interactive web interface allowing users to input smartwatch specifications and get real-time price recommendations.
- [ ] **Feature Selection / Dimensionality Reduction**: Group rare `Model Name` high-cardinality categories to reduce one-hot column expansion from 204 to ~30 dense features.
- [ ] **Ensemble Stacking**: Implement a Stacking Regressor combining Ridge Regression, LightGBM, and XGBoost for enhanced variance reduction.
- [ ] **Dockerization & CI/CD**: Containerize the inference API with automated unit testing and GitHub Actions.

---

## 📄 License & Attribution

This project is part of the **ML Mini Practice Projects** collection by [Bhavesh Solanki](https://github.com/bhaveshsolanki159-codex). Open-sourced under the [MIT License](https://opensource.org/licenses/MIT).
