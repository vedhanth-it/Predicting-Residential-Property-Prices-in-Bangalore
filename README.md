# 🏠 Predicting Residential Property Prices in Bangalore

## 📌 Problem Statement
The goal of this project is to **predict residential property prices in Bangalore** based on various housing attributes such as location, size, total square feet, number of bathrooms, and balconies.  
This helps real estate companies and buyers make informed price decisions.

---

## 🧩 Dataset
- **File:** `Bangalore house data.csv`
- **Shape:** Displayed in notebook using `df.shape`
- **Features:**
  - `location`
  - `size`
  - `total_sqft`
  - `bath`
  - `balcony`
  - `price` *(target variable)*

---

## 🔍 Exploratory Data Analysis (EDA)
Performed to understand data patterns and distributions:
- Checked for **missing values** and **data types**
- Visualized relationships:
  - `price` vs `total_sqft`
  - `price` vs `location`
  - `price` vs `size`
- Detected and removed **outliers** using **IQR** and **Z-score**
- Encoded categorical features (`location`) using **One-Hot Encoding**

**Key Insights:**
- Property prices per sqft vary across different locations.
- Larger homes in premium areas show high price variability.
- Outlier removal improved model accuracy.

---

## 🧮 Feature Engineering
- Extracted numeric values from `size` and `total_sqft`
- Created new feature: `price_per_sqft`
- Applied **StandardScaler** and **MinMaxScaler**
- Encoded categorical variables
- Removed redundant or highly correlated features

---

## 🤖 Models Used
Trained and compared multiple regression models:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Random Forest Regressor**
5. **Gradient Boosting Regressor**
6. **XGBoost Regressor**
7. **Support Vector Regressor (SVR)**

---

## ⚙️ Model Training & Evaluation
- Split dataset into **Train (80%)** and **Test (20%)**
- Used **K-Fold Cross Validation (k=5)**
- Hyperparameter tuning via **RandomizedSearchCV** and **GridSearchCV**

**Metrics Used:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📈 Model Performance Summary

| Model | R² Score | RMSE | MAE |
|--------|-----------|------|------|
| Linear Regression | ~0.78 | Moderate | Moderate |
| Ridge/Lasso Regression | ~0.80 | Lower | Lower |
| Random Forest | ~0.88 | Good | Good |
| Gradient Boosting | ~0.90 | Better | Better |
| **XGBoost** | **~0.92** | **Lowest RMSE** | **Lowest MAE** |

✅ **Best Model:** **XGBoost Regressor**

---

## 📊 Visualizations
Included visual analyses:
- Feature distributions
- Correlation heatmap
- Price vs Square Feet scatterplots
- Feature importance for tree-based models

---

## 🧠 Conclusions
- **XGBoost** achieved the highest prediction accuracy.
- **Location** and **Total Square Feet** are key predictors.
- Removing outliers and engineering meaningful features greatly improved results.
- The final model can serve as a baseline for **real estate price prediction systems**.

---

## 🚀 Future Work
- Integrate additional features (e.g., year built, amenities)
- Use **geocoordinates** instead of location strings
- Deploy the trained model using **Flask** or **Streamlit** web apps

---

## 🧾 Dependencies
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost


