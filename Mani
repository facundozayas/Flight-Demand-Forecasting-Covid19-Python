# Flight Demand Forecasting Post COVID-19 ✈️

**Type:** Data Science Project / Forecasting  
**Author:** Facundo Zayas  
**Tools:** Python, Pandas, Matplotlib, Scikit-learn, Jupyter Notebook  

---

## 📋 Project Overview

This project analyzes and forecasts flight demand in the post-COVID-19 period.  
The main goal was to explore how the pandemic affected passenger volume and to model the expected recovery trend in the airline industry.

The analysis combines **exploratory data analysis (EDA)**, **visualization**, and **machine learning models** to predict flight demand levels over time.

---

## 🎯 Objectives

- Understand the impact of COVID-19 on flight demand.  
- Identify key variables affecting recovery (routes, seasons, restrictions).  
- Build predictive models for future flight volumes.  
- Visualize trends and compare pre- and post-pandemic patterns.

---

## 🧠 Methodology

1. **Data Preparation**
   - Cleaned and formatted flight volume data from multiple periods.  
   - Handled missing values and normalized time indexes.  

2. **Exploratory Data Analysis**
   - Time series visualization of total flights per month.  
   - Correlation between restrictions, holidays, and demand recovery.  

3. **Modeling**
   - Applied regression models (e.g., Linear Regression, Random Forest).  
   - Evaluated using metrics such as RMSE and R².  

4. **Forecasting**
   - Predicted short-term demand under post-COVID recovery assumptions.  
   - Compared actual vs. predicted flight volumes.

---

## 📊 Key Insights

- Demand dropped sharply during 2020, reaching less than 10% of pre-COVID levels.  
- Recovery showed a **seasonal pattern**, starting to stabilize mid-2022.  
- The **model successfully captured** recovery trends with moderate prediction error.  
- External factors (restrictions, oil prices) remain major demand drivers.

---

## 🔢 Example Code

```python
# Load and prepare data
import pandas as pd
df = pd.read_csv("flight_demand.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').asfreq('M')

# Split data
train = df[:'2022']
test = df['2023':]

# Train a simple model
from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.arange(len(train)).reshape(-1, 1)
y_train = train['Flights']

model = LinearRegression()
model.fit(X_train, y_train)
