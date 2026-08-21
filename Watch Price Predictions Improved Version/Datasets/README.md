# 📊 Smartwatch Pricing Datasets & Data Dictionary

This directory contains the raw e-commerce smartwatch dataset and the engineered tabular dataset utilized across the machine learning modeling lifecycle.

---

## 📁 Directory Structure

```text
Datasets/
├── smartwatches.csv    # Raw dataset (450 rows × 16 columns)
├── cleaned.csv         # Feature engineered dataset (181 rows × 204 columns)
└── README.md           # Dataset documentation and data dictionary
```

---

## 1. Raw Dataset: `smartwatches.csv`

The raw dataset contains 450 product listings scraped from Indian e-commerce platforms.

### Data Dictionary

| Column Name | Data Type | Null Count | Description | Example Values |
| :--- | :---: | :---: | :--- | :--- |
| `Unnamed: 0` | Integer | 0 | Raw row index | `0`, `1`, `2` |
| `Brand` | String | 0 | Smartwatch brand / manufacturer | `noise`, `fire-boltt`, `boAt`, `apple` |
| `Current Price` | Float | 0 | Current retail / selling price in INR (₹) | `82990.0`, `3799.0`, `1999.0` |
| `Original Price` | Float | 0 | Maximum Retail Price (MRP) in INR (₹) | `89900.0`, `19999.0`, `5990.0` |
| `Discount Percentage` | Float | 0 | Listed discount percentage | `7.0`, `81.0`, `66.0` |
| `Rating` | Float | 59 | Average customer review rating out of 5.0 | `4.2`, `4.5`, `3.9` |
| `Number OF Ratings` | Float | 59 | Total count of customer reviews / ratings | `1250.0`, `8430.0`, `45.0` |
| `Model Name` | String | 35 | Model name or product title | `ColorFit Pro 4`, `Ninja Call 2` |
| `Dial Shape` | String | 123 | Physical shape of the smartwatch face | `Square`, `Circle`, `Rectangle` |
| `Strap Color` | String | 30 | Primary color of the smartwatch band | `Black`, `Blue`, `Grey`, `Pink` |
| `Strap Material` | String | 69 | Material used for the strap band | `Silicon`, `Stainless Steel`, `Leather` |
| `Touchscreen` | String | 37 | Indicates whether screen is touch-enabled | `Yes`, `No` |
| `Battery Life (Days)`| Float | 26 | Manufacturer claimed battery life in days | `7.0`, `10.0`, `1.5`, `14.0` |
| `Bluetooth` | String | 2 | Indicates Bluetooth connectivity / calling | `Yes`, `No` |
| `Display Size` | String | 200 | Screen diagonal dimension string with units | `"1.4 inches"`, `"1.8 inches"`, `"1.69 inch"` |
| `Weight` | String | 186 | Product weight range string | `"20 - 35 g"`, `"35 - 50 g"`, `"75g +"`, `"<= 20 g"` |

---

## 2. Processed & Engineered Dataset: `cleaned.csv`

The cleaned dataset is produced by running [`feature_engineering.ipynb`](file:///c:/Users/bhave/OneDrive/Desktop/ML%20Projects/Watch%20Price%20Predictions%20Improved%20Version/feature_engineering.ipynb). It contains **181 cleaned samples** and **204 transformed numerical / one-hot encoded features**.

### Key Transformations Applied:

1. **Target Feature Engineering**:
   - `Discount Price` calculated as: $\text{Original Price} \times \frac{-\text{Discount Percentage}}{100}$
   - Dropped the redundant `Discount Percentage` and raw indexing columns.

2. **Regex Parsing & Unit Standardization**:
   - `Display Size`: Extracted float values from strings (e.g., `"1.8 inches"` $\rightarrow `1.8`$).
   - `Weight`: Mapped string ranges to numeric midpoints:
     - `"20 - 35 g"` $\rightarrow `27.5`$
     - `"35 - 50 g"` $\rightarrow `42.5`$
     - `"50 - 75 g"` $\rightarrow `62.5`$
     - `"75g +"` $\rightarrow `75.0`$
     - `"<= 20 g"` $\rightarrow `20.0`$

3. **Multivariate IQR Outlier Filtering**:
   - Outliers in `['Current Price', 'Original Price', 'Rating', 'Number OF Ratings', 'Display Size']` were removed using the $1.5 \times \text{IQR}$ threshold rule:
     $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
   - Filtered dataset from 450 down to 181 consistent, non-skewed observation points.

4. **Hierarchical Brand-Level Imputation**:
   - Continuous attributes (`Display Size (Inches)`, `Weight (g)`, `Battery Life (Days)`, `Rating`, `Number OF Ratings`) filled with brand-level medians:
     ```python
     brand_median = train_df.groupby('Brand')[col].transform('median')
     train_df[col] = train_df[col].fillna(brand_median).fillna(train_df[col].median())
     ```
   - Categorical attributes (`Dial Shape`, `Strap Material`, `Touchscreen`, `Bluetooth`, `Model Name`) filled with `'Unknown'`.

5. **Min-Max Scaling**:
   - Scaled continuous features into range $[0, 1]$ using `sklearn.preprocessing.MinMaxScaler`:
     - `Current Price`, `Original Price`, `Rating`, `Number OF Ratings`, `Battery Life (Days)`, `Display Size (Inches)`, `Weight (g)`.

6. **One-Hot Encoding**:
   - Applied `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` across:
     - `Brand`, `Model Name`, `Dial Shape`, `Strap Material`, `Touchscreen`, `Bluetooth`.

---

## 3. Quick Data Loading Example

```python
import pandas as pd

# Load raw dataset
raw_df = pd.read_csv('Datasets/smartwatches.csv', index_col=0)
print(f"Raw Data Dimensions: {raw_df.shape}")

# Load engineered dataset for modeling
cleaned_df = pd.read_csv('Datasets/cleaned.csv')
print(f"Cleaned Data Dimensions: {cleaned_df.shape}")

X = cleaned_df.drop(columns=['Discount Price'])
y = cleaned_df['Discount Price']
```
