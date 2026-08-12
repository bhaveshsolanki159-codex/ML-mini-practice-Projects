# Ecommerce Customers — Linear Regression

A machine learning project that uses **Linear Regression** to predict a customer's **Yearly Amount Spent** based on their interaction with an e-commerce platform.

## Project Overview

This project explores customer behavior and builds a Linear Regression model using the following features:

- Average Session Length
- Time on App
- Time on Website
- Length of Membership

The target variable is:

- **Yearly Amount Spent**

The project includes exploratory data analysis (EDA), model training, predictions, evaluation metrics, and residual analysis.

## Dataset

The notebook expects a CSV file named:

```text
Ecommerce_Customers.csv
```

The dataset contains customer information, including numerical and categorical variables. The notebook performs basic checks for:

- Dataset dimensions
- Missing/null values
- Duplicate rows
- Data types
- Descriptive statistics

## Exploratory Data Analysis

The notebook uses Seaborn and Matplotlib to investigate relationships in the data.

The analysis includes:

- Boxplots for numerical variables
- Joint plots
- Pair plots
- Correlation/relationship analysis between customer behavior and yearly spending

The exploratory analysis suggests that **Time on App**, **Average Session Length**, and **Length of Membership** have useful relationships with yearly spending.

## Machine Learning Workflow

The project follows this workflow:

1. Load the dataset using Pandas.
2. Inspect and clean the data.
3. Perform exploratory data analysis.
4. Select features and target variable.
5. Split the dataset into training and testing sets.
6. Train a Linear Regression model.
7. Generate predictions.
8. Evaluate model performance.
9. Analyze residuals.

### Features

```text
Avg. Session Length
Time on App
Time on Website
Length of Membership
```

### Target

```text
Yearly Amount Spent
```

The data is split using:

```python
train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
```

## Model

The project uses `LinearRegression` from scikit-learn.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
```

The learned coefficients are displayed in a DataFrame to understand how each feature contributes to the prediction.

## Model Evaluation

The model is evaluated using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

A prediction-vs-actual scatter plot is also used to visually evaluate the model.

## Residual Analysis

Residuals are calculated as:

```python
residuals = y_test - predictions
```

The notebook examines the residuals using:

- Residual distribution
- Histogram/KDE plot
- Normal probability (Q-Q) plot

These plots help assess whether the errors are reasonably distributed and whether the Linear Regression assumptions are appropriate.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
venv/Scripts/activate
```

Install the required packages:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

> Note: The package is installed as `scikit-learn`, but imported in Python as `sklearn`.

## Running the Project

1. Clone or download the project.
2. Place `Ecommerce_Customers.csv` in the project directory.
3. Create and activate the virtual environment.
4. Install the dependencies.
5. Open the notebook:

```powershell
jupyter notebook
```

6. Open `main.ipynb`.
7. Run the notebook cells from top to bottom.

## Project Structure

```text
Linear Regression Project/
│
├── main.ipynb
├── Ecommerce_Customers.csv
├── venv/
└── README.md
```

## Key Learning Outcomes

This project demonstrates:

- Loading and inspecting a dataset with Pandas
- Exploratory Data Analysis
- Feature and target selection
- Train/test splitting
- Building a Linear Regression model
- Understanding regression coefficients
- Making predictions
- Evaluating regression models using MAE, MSE, and RMSE
- Performing residual analysis

## Future Improvements

Possible improvements include:

- Feature scaling and comparison with other regression models
- Cross-validation
- R² score evaluation
- Checking multicollinearity using VIF
- More detailed feature engineering
- Comparing Linear Regression with models such as Random Forest or Gradient Boosting
- Creating a reusable prediction script or web application
