<p align="center">
  <h1 align="center">⌚ Smartwatch Price Prediction</h1>
  <p align="center">
    <em>A machine learning project to predict smartwatch prices based on brand, specifications, and market features.</em>
  </p>
  <p align="center">
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-dataset">Dataset</a> •
    <a href="#-methodology">Methodology</a> •
    <a href="#-results">Results</a> •
    <a href="#-tech-stack">Tech Stack</a>
  </p>
</p>

---

## 📌 Problem Statement

The smartwatch market has seen explosive growth, with prices ranging from ₹1,199 to ₹1,39,990 across dozens of brands. With such a wide range, it becomes crucial for both consumers and retailers to understand what drives pricing.

**Goal:** Build a predictive model that estimates the current selling price of a smartwatch based on its brand, hardware specifications (display size, battery life, weight), features (touchscreen, Bluetooth), and market signals (original price, discount percentage, user ratings).

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | `smartwatches.csv` |
| **Records** | 450 smartwatches |
| **Features** | 15 columns |
| **Brands** | 18 unique brands |

### Features

| Feature | Type | Description |
|---|---|---|
| `Brand` | Categorical | Manufacturer (Apple, Samsung, Boat, Fire-Boltt, Noise, etc.) |
| `Current Price` | Numerical | Current selling price (₹) |
| `Original Price` | Numerical | Listed MRP (₹) |
| `Discount Percentage` | Numerical | Discount offered (%) |
| `Rating` | Numerical | User rating (1–5 scale) |
| `Number OF Ratings` | Numerical | Total number of user ratings |
| `Model Name` | Categorical | Model identifier |
| `Dial Shape` | Categorical | Shape of the watch dial |
| `Strap Color` | Categorical | Color of the strap |
| `Strap Material` | Categorical | Material type (Silicon, Leather, Metal, etc.) |
| `Touchscreen` | Categorical | Touchscreen support (Yes/No) |
| `Battery Life (Days)` | Numerical | Battery life in days |
| `Bluetooth` | Categorical | Bluetooth support (Yes/No) |
| `Display Size` | Numerical | Display size in inches |
| `Weight` | Categorical | Weight range category |

### Brands Covered

```
amazfit • ambrane • apple • boat • crossbeats • dizo • fire-boltt • fitbit
fossil • garmin • gizmore • hammer • honor • huawei • noise • pebble • samsung • zebronics
```

---

## 🔬 Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

### 1. 🧠 Problem Identification & Business Understanding
- Defined the prediction target: **Current Price** of smartwatches
- Identified key business question: *What specifications and market signals most influence smartwatch pricing?*

### 2. 📥 Data Collection
- Collected smartwatch listing data with 450 records across 18 brands
- Features span hardware specs, market pricing, and user feedback

### 3. 🧹 Data Pre-Processing & Cleaning

#### Missing Value Treatment
- **Categorical columns:** Imputed using the **mode** (most frequent value) grouped by `Brand`
- **Numerical columns:** Imputed using the **median** grouped by `Brand`
- This brand-aware imputation preserves the natural distribution within each brand's product line

#### Data Type Corrections
- Converted `Display Size` from string (e.g., `"1.8 inches"`) → float by stripping unit suffixes

#### Duplicate Removal
- Identified and removed duplicate records, then reset the index

#### Data Validation
- Built a custom `check()` function to audit every column for:
  - Data type
  - Valid instance count
  - Unique values
  - Null count
  - Duplicate count

### 4. 📈 Outlier Detection
- Applied the **IQR (Interquartile Range) method** on all numerical columns
- Generated summary statistics using `tabulate` for clean reporting
- Created **interactive box plots** using `Plotly Express` for visual inspection
- Identified outliers in columns like `Current Price`, `Original Price`, `Number OF Ratings`, and `Display Size`

### 5. 🤖 Data Modeling *(In Progress)*
> Feature engineering, encoding, and model training coming next.

### 6. 📊 Model Evaluation *(Upcoming)*
> Model comparison, cross-validation, and performance metrics.

### 7. 🚀 Model Deployment *(Upcoming)*
> Deployment pipeline and serving predictions.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.13 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly Express |
| **Reporting** | Tabulate |
| **Environment** | Jupyter Notebook, venv |
| **Version Control** | Git & GitHub |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ installed
- `pip` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/bhaveshsolanki159-codex/ML-mini-practice-Projects.git
cd "ML-mini-practice-Projects/Watch Price Predictions"

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly tabulate jupyter
```

### Run the Notebook

```bash
jupyter notebook main.ipynb
```

---

## 📁 Project Structure

```
Watch Price Predictions/
│
├── main.ipynb          # Main Jupyter notebook with all analysis & modeling
├── smartwatches.csv    # Raw dataset (450 records × 15 features)
├── README.md           # Project documentation (you are here)
├── venv/               # Python virtual environment
└── .gitignore          # Git ignore rules
```

---

## 📈 Key Findings (So Far)

- **Price distribution** is heavily right-skewed — most smartwatches fall in the ₹1,200–₹5,000 range, with premium outliers (Apple, Garmin, Fossil) exceeding ₹80,000
- **Discount percentages** range from -79% (markup) to 91%, with a mean of ~48%
- **Battery life** clusters around 8 and 17–22 days, suggesting two distinct product categories (fitness bands vs. full smartwatches)
- **User ratings** are tightly distributed (mean 4.03, std 0.55), indicating overall positive reception across brands
- **Significant outliers** were detected in `Current Price`, `Original Price`, and `Number OF Ratings` columns

---

## 🗺️ Roadmap

- [x] Problem identification & business understanding
- [x] Data collection
- [x] Data pre-processing (missing values, type conversion, deduplication)
- [x] Outlier detection & visualization
- [ ] Exploratory Data Analysis (EDA) — correlation, distribution plots, brand-wise analysis
- [ ] Feature engineering & encoding
- [ ] Model training (Linear Regression, Random Forest, XGBoost, etc.)
- [ ] Model evaluation & hyperparameter tuning
- [ ] Model deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open source and available for educational purposes.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/bhaveshsolanki159-codex">Bhavesh Solanki</a>
</p>