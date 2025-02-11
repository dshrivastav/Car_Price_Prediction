# 🚗 Car Price Prediction Using Machine Learning  

## 📌 Overview  
This project leverages machine learning to predict used car prices based on publicly available data.  
By analyzing factors like mileage, age, manufacturer, and fuel type, we develop a predictive model  
that provides insights into price determinants and recommends optimal selling strategies.
The project focuses on predicting vehicle prices using various regression models, including Linear Regression, 
Polynomial Regression (degrees 2, 3, 4), Ridge Regression, and Lasso Regression. 
The dataset contains features such as the vehicle's price, year, mileage, manufacturer, fuel type, and other attributes.

We apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset, enhance model performance, and visualize the impact of model complexity through various evaluations.  

---
## Objective:
The objective of this analysis is to:
Understand the relationships between vehicle features and their prices.
Build multiple regression models to predict vehicle prices.
Evaluate and compare the performance of different regression models.

## Requirements
To run this project, you need to install the following Python packages:
pandas
numpy
scikit-learn
matplotlib
seaborn

## Dataset
The dataset used in this analysis contains the following columns:

id: Unique vehicle identifier
region: The region where the vehicle is located
price: The price of the vehicle (target variable)
year: The year the vehicle was manufactured
manufacturer: The vehicle's manufacturer
model: The model of the vehicle
condition: The condition of the vehicle (e.g., new, used)
cylinders: Number of cylinders
fuel: Type of fuel used by the vehicle (e.g., gasoline, diesel)
odometer: The vehicle's mileage
title_status: Vehicle title status
transmission: Type of transmission (automatic, manual)
VIN: Vehicle Identification Number
drive: Drive type (e.g., AWD, FWD)
size: The size category of the vehicle
type: The vehicle type (e.g., SUV, sedan)
paint_color: Color of the vehicle
state: The state where the vehicle is located


## 📊 Data Exploration & Preprocessing

### Analysis Steps
1. Data Preparation:

Handling missing values (imputation and removal).
Feature encoding for categorical variables (e.g., manufacturer, region).
Outlier detection and capping for price and odometer.
Standardization of numerical features (price, odometer, year).

2. Principal Component Analysis (PCA):

Applied PCA to reduce dimensionality while retaining 95% of the variance in the data.
This step helps improve model performance and reduces overfitting.

3. Model Training:

Linear Regression: A simple linear model to predict the vehicle prices.
Polynomial Regression: Polynomial features (degrees 2, 3, 4) to capture non-linear relationships.
Ridge Regression: Linear regression with L2 regularization to avoid overfitting.
Lasso Regression: Linear regression with L1 regularization, performing automatic feature selection.

4. Model Evaluation:

Models were evaluated using MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R² (R-squared) to compare performance.
We also used Validation Curves and Learning Curves to analyze the bias-variance tradeoff and the model's performance with different data sizes.

### 🔍 Key Steps:  
✅ **Dropped irrelevant columns** (e.g., ID, VIN, and highly missing columns like "size").  
✅ **Handled missing values** using imputation (median for numerical, mode for categorical).  
✅ **Encoded categorical variables** with a combination of One-Hot Encoding & James-Stein Encoding.  
✅ **Standardized numerical features** to improve model performance.  

### 📂 Dataset Features:  
- **Numerical**: Year, odometer, price, cylinders, fuel efficiency, etc.  
- **Categorical**: Manufacturer, type, transmission, fuel, drive, etc.  

---

## 🚀 Model Training & Comparison  
| Model | RMSE (Lower is better) | Observations |
|--------|------|------|
| **Linear Regression** | 4500+ | Struggled with non-linearity |
| **Ridge Regression** | 4300+ | Slightly better than Linear Regression |
| **Lasso Regression** | 4400+ | Performed worse than Ridge |
| **Random Forest** | **3100** | Strong performance, captures non-linearity |
| **Gradient Boosting** | **2900** | Best performance after tuning |

✅ **Final Model Selected:** **Gradient Boosting Regressor**  

---

### 📌 Feature Importance Analysis  
#### 🔑 Top Factors Influencing Car Prices:  
📌 **1. Odometer (Mileage) - Strongest Predictor** 🚗   
   - Lower mileage = Higher price.  
📌 **2. Vehicle Age (Year) - Key Factor** 📆  
   - Newer cars have higher resale value.  
📌 **3. Manufacturer & Model - Price Varies by Brand** 🏷️  
   - Luxury brands (BMW, Mercedes) retain value better.  
📌 **4. Condition & Fuel Type Impact Pricing** ⚡  
   - Electric & hybrid cars are priced higher.  
📌 **5. Drive Type & Transmission - AWD/4WD Add Value** 🛞  
   - More valuable in snowy/off-road regions.

### Model Deployment (Optional)
You can deploy the trained model using Flask or FastAPI. 

## Conclusion
The models evaluated include Ridge and Lasso regression. The models provide valuable insights into how various features influence car prices, such as:

Year: Newer cars tend to have higher prices.
Odometer: Higher mileage decreases the price.
Manufacturer: Certain manufacturers tend to have higher price points.
Future Improvements
Explore non-linear models such as Random Forests or Gradient Boosting Machines.
Incorporate additional features like car condition or historical sales data for better accuracy.

As part of our project to predict used car prices, we’ve applied several regression models to understand the key drivers of car prices. The goal of this work is to provide actionable insights that can help used car dealers fine-tune their inventory and optimize pricing strategies. This report outlines the main findings from the modeling process, including which features drive car prices, the quality of our models, and how these insights can be used to adjust the pricing strategy.

### Key Findings
#### 1. Model Performance and Evaluation
We explored various models to predict used car prices based on available features, such as car age, mileage, and manufacturer. The key models used in this analysis were Ridge regression and Lasso regression.

Ridge Regression: This model performed well and was able to predict car prices with reasonable accuracy. It helped identify which features (such as mileage, year, and manufacturer) are important in setting car prices.

Lasso Regression: Lasso also performed well but with a key advantage: it effectively selected important features while driving irrelevant features to zero. This feature selection process helps in simplifying the model by focusing on the most influential features.

Polynomial Regression: We also tested a polynomial regression model (degree 3), but it did not provide significant improvements over Ridge and Lasso. This suggests that the relationships between features and car prices are mostly linear in nature for this dataset.

#### 2. Key Drivers of Used Car Prices
From the model results, we were able to identify key drivers of used car prices. These drivers are the features that most influence how much a car is worth on the market:

Year (Age of the Car): Newer cars tend to have higher prices. The age of the vehicle is a strong determinant of its price, with newer models generally valued higher.

Odometer (Mileage): As expected, cars with lower mileage are valued higher. High mileage negatively impacts the price of a used car.

Manufacturer: The brand or manufacturer of a car significantly impacts its price. Some manufacturers have higher demand and perceived value, leading to higher prices.

#### 3. Model Quality and Predictive Power
The models performed well, with R² values indicating that both Ridge and Lasso regression models explain a significant portion of the variance in car prices. Both models were evaluated on their ability to predict prices on the test set, showing that they generalize well and perform reliably on unseen data.

**R² for Ridge:** Measures how well the model fits the training and test data. A high R² value on both training and test sets indicates a good model fit.
**Residual Variance:** The residual variance (difference between actual and predicted values) was low, indicating that the models captured the underlying patterns in the data well.
## Actionable Insights for Used Car Dealers
Based on the analysis and model findings, here are actionable insights for fine-tuning your inventory and optimizing pricing strategies:

Pricing for Newer Cars: Focus on newer vehicles in your inventory, as they tend to have a higher price. Dealers can use the model to predict the price of new cars and ensure they are not underpricing valuable inventory.

Impact of Mileage: For cars with high mileage, dealers should consider offering discounts or pricing them more competitively. High-mileage cars tend to have lower prices, and dealers can use this insight to set more accurate pricing and avoid overpricing vehicles with higher usage.

Brand or Manufacturer Considerations: Focus on cars from high-demand manufacturers. Certain brands, due to their perceived reliability or brand value, may warrant higher prices. Understanding which manufacturers contribute most to price can help dealers adjust their inventory to meet market demand.

Optimizing Inventory Based on Model Predictions: By applying this model to the entire inventory, dealers can forecast the optimal price for each car, identify overpriced or underpriced cars, and make adjustments accordingly. This can help increase sales and reduce unsold inventory.
