<p align="center">
  <h1 align="center">🧠 ML Mini Practice Projects</h1>
  <p align="center">
    <em>A collection of hands-on, end-to-end machine learning practice projects built to master core ML concepts, exploratory data analysis, feature engineering, and predictive modeling through real-world implementations.</em>
  </p>
  <p align="center">
    <a href="#-about">About</a> •
    <a href="#-projects">Projects</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-roadmap--concepts-to-explore-next">Roadmap</a>
  </p>
</p>

---

## 📖 About

This repository is a **personal learning lab** — each project focuses on specific machine learning concepts and applies them to real-world datasets. The goal is to **deeply understand the fundamentals** by getting hands dirty with data ingestion, auditing, feature transformation, model selection, hyperparameter tuning, and evaluation.

**Philosophy:** *Learn by building. One concept at a time.*

Each project is self-contained with its own dataset, notebooks, virtual environment configurations, and documentation.

---

## 🗂️ Projects

| # | Project | ML Concepts & Techniques | Dataset | Status |
|---|---------|--------------------------|---------|--------|
| 1 | [Linear Regression Project](./Linear%20Regression%20Project/) | Linear Regression, Residual Analysis, Q-Q Plots | E-commerce Customers | ✅ Complete |
| 2 | [Watch Price Predictions](./Watch%20Price%20Predictions/) | CRISP-DM, Brand-Aware Imputation, IQR Outliers, Multi-Feature EDA | Smartwatches (450 records) | ✅ Complete |
| 3 | [Watch Price Predictions Improved Version](./Watch%20Price%20Predictions%20Improved%20Version/) | Regex Feature Extraction, Hierarchical Imputation, One-Hot Encoding, XGBoost Tuning (GridSearchCV), 5-Fold CV, Pipeline Serialization (`model.pkl`) | Smartwatches (181 cleaned rows × 204 features) | ✅ Complete |

---

### 1. 📈 Linear Regression — E-commerce Customer Spending

Predicts a customer's **Yearly Amount Spent** on an e-commerce platform using behavioral features such as session length, time on mobile app, time on website, and membership duration.

**Key Learnings & Techniques:**
- Exploratory Data Analysis with Seaborn jointplots and pairplots
- Train/test splitting and baseline fitting with Scikit-learn
- Regression performance evaluation metrics (MAE, MSE, RMSE)
- Residual analysis and normality verification (Histograms, KDE, Q-Q plots)
- Interpreting linear regression coefficients to guide business recommendations

➡️ [View Project Documentation & Code](./Linear%20Regression%20Project/README.md)

---

### 2. ⌚ Watch Price Predictions — Smartwatch Pricing & Exploratory Analysis

Analyzes smartwatch specifications, brand positioning, and pricing dynamics across 18 brands to understand market drivers and predict selling prices.

**Key Learnings & Techniques:**
- CRISP-DM framework for structured machine learning workflows
- Custom data auditing utilities (`check`, `check_unique`) for missing values and duplicates
- Brand-aware grouped imputation (grouped mode for categorical, grouped median for numerical)
- IQR-based outlier detection with interactive Plotly Express box plots
- Multi-dimensional exploratory data analysis on battery life tradeoffs, brand equity, and discounting patterns

➡️ [View Project Documentation & Code](./Watch%20Price%20Predictions/README.md)

---

### 3. 🚀 Smartwatch Price Prediction & ML Pipeline (Improved Version)

A modular, production-grade machine learning pipeline predicting smartwatch discount pricing with automated feature extraction, rigorous cross-validation, hyperparameter tuning, and model persistence.

**Key Learnings & Techniques:**
- Advanced regex string parsing for numerical conversions (e.g. `Display Size` and `Weight` range midpoints)
- Multivariate IQR outlier filtering and hierarchical imputation (Brand median with fallback to Global median)
- High-cardinality categorical one-hot encoding and min-max scaling
- Multi-model evaluation comparing Linear Regression, Decision Trees, Random Forest Regressors, and XGBoost
- 5-Fold cross-validation and exhaustive `GridSearchCV` optimization (achieving $R^2 = 0.8164$)
- Encapsulated Scikit-Learn `Pipeline` serialized to `model.pkl` for one-line inference

➡️ [View Project Documentation & Code](./Watch%20Price%20Predictions%20Improved%20Version/README.md) | 📊 [Dataset Documentation](./Watch%20Price%20Predictions%20Improved%20Version/Datasets/README.md)

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ / 3.13 |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn, Plotly Express |
| **Machine Learning & Preprocessing** | Scikit-learn, XGBoost, SciPy |
| **Data Auditing & Profiling** | Tabulate, ydata-profiling |
| **Environment & Tooling** | Jupyter Notebook, JupyterLab, Python `venv` |
| **Version Control** | Git & GitHub |

---

## 🚀 Getting Started

Each project is self-contained. Navigate into any project folder and follow its instructions:

```bash
# Clone the repository
git clone https://github.com/bhaveshsolanki159-codex/ML-mini-practice-Projects.git
cd ML-mini-practice-Projects

# Navigate to a project of your choice
cd "Linear Regression Project"
# or
cd "Watch Price Predictions"
# or
cd "Watch Price Predictions Improved Version"

# Create a virtual environment & activate it
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# (or install project-specific packages listed in its README)

# Launch Jupyter Notebook or JupyterLab
jupyter lab
# or
jupyter notebook main.ipynb
```

---

## 🗺️ Roadmap — Concepts to Explore Next

- [x] Simple & Multiple Linear Regression
- [x] Residual & Normality Analysis
- [x] Structured Data Cleaning & Hierarchical Imputation
- [x] Decision Trees & Random Forest Regressors
- [x] Gradient Boosting & XGBoost Regressors
- [x] Hyperparameter Tuning with GridSearchCV & Cross-Validation
- [ ] Logistic Regression (Classification)
- [ ] K-Nearest Neighbors (KNN)
- [ ] Support Vector Machines (SVM)
- [ ] K-Means & Hierarchical Clustering
- [ ] Principal Component Analysis (PCA) & Dimensionality Reduction
- [ ] Natural Language Processing (NLP) basics
- [ ] Deep Learning & Neural Networks with PyTorch / TensorFlow
- [ ] Interactive Web Apps (Streamlit / FastAPI deployment)

---

## 🤝 Contributing

This is a personal learning repository, but suggestions, feedback, and improvements are always welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/bhaveshsolanki159-codex">Bhavesh Solanki</a>
</p>
