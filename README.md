# Car Price Prediction Using Machine Learning  

## Overview  
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


## Data Exploration & Preprocessing

### Analysis Steps
**1. Data Preparation:**

### Dataset Features:  
- **Numerical**: Year, odometer, price, cylinders, fuel efficiency, etc.  
- **Categorical**: Manufacturer, type, transmission, fuel, drive, etc.
  
**Handling missing values (imputation and removal).**
Some decisions for data preparation step
**Keep:**
price: Target variable; essential for prediction. year, odometer: Crucial for understanding the vehicle's value and condition. Keep both region and state based on the unique values

**Drop:**
size: High percentage of missing data, which makes it less useful. If the missingness is too high (e.g., 50%+), it should be removed.
ID and VIN: As a unique identifier, it doesn’t add value for modeling and can safely be dropped.
region Has redundancy with state. Both of these features have too many unique values, causing high cardinality in encoding, Choosing to drop region.

**Drop (after encoding):**
manufacturer, model, fuel, drive, state, and region are all categorical variables that can be encoded (James-Stein for manufacturer and region, and One-Hot Encoding for others). After encoding, these features are no longer required in their original form and can be dropped.

**Drop redundant columns:**
Consider dropping cylinders, color, title_status: While they might seem relevant, these features are not as significant for predicting price compared to the other factors like year and odometer. Additionally, they are either too specific or may have a lot of missing or irrelevant data.

<img width="627" alt="Screenshot 2025-02-11 at 1 59 17 PM" src="https://github.com/user-attachments/assets/2c7fdb65-3dba-4e0e-8da7-2c0e44e7d197" />


**Feature encoding for categorical variables (e.g., manufacturer, region).**
1. James-Stein (Target) Encoding for 'manufacturer' columns
2. One Hot Encoding for fuel', 'transmission', 'title_status', 'condition', 'cylinders'
3. Target Encoding to 'model', 'drive', 'type', 'paint_color', 'state'
   
**Outlier detection and capping for price and odometer.**
Here’s a summary of key numeric columns:
price: Mean: ~75,199 Min: 0 (there might be errors or data anomalies) Max: ~$3.7 billion (likely outliers) 
year: Mean: 2011 Min: 1900 (likely some data issues) Max: 2022 
odometer: Mean: 98,043 miles Min: 0 (could indicate new or erroneous data) Max: 10 million miles (outlier alert!) 
We handled these outliers
<img width="1046" alt="Screenshot 2025-02-11 at 1 59 02 PM" src="https://github.com/user-attachments/assets/04cea673-34bf-42b5-9037-24b11c0e74ce" />

**Standardization of numerical features (price, odometer, year).**

**2. Principal Component Analysis (PCA)**

Applied PCA to reduce dimensionality while retaining 95% of the variance in the data.
This step helps improve model performance and reduces overfitting.

**3. Model Training:**

Linear Regression: A simple linear model to predict the vehicle prices.
Polynomial Regression: Polynomial features (degrees 2, 3, 4) to capture non-linear relationships.
Ridge Regression: Linear regression with L2 regularization to avoid overfitting.
Lasso Regression: Linear regression with L1 regularization, performing automatic feature selection.
Gradient Boosting: A tree-based model capable of capturing complex non-linear relationships between features.

Train and evaluate multiple models: Linear Regression, Polynomial Regression (degree 2, 3, 4) using PCA-transformed features.
Evaluate the models based on MSE, RMSE, and R².

**4. Model Evaluation:**

Models were evaluated using MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R² (R-squared) to compare performance.
We also used Validation Curves and Learning Curves to analyze the bias-variance tradeoff and the model's performance with different data sizes.
Through the modeling process, we explored multiple regression techniques to predict used car prices. Here's a summary of key insights from the models:

**Best Ridge Alpha: 1000
Best Lasso Alpha: 0.01
Best Polynomial Degree: 3
Gradient Boosting Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
Gradient Boosting Best RMSE: 0.17**

**Ridge Regression  MSE: 0.033568, RMSE: 0.183217, R²: 0.453059
Lasso Regression  MSE: 0.033568, RMSE: 0.183216, R²: 0.453059
Polynomial (deg 3)  MSE: 0.032793, RMSE: 0.181089, R²: 0.465686
Gradient Boosting  MSE: 0.031717, RMSE: 0.178092, R²: 0.483227**

<img width="700" alt="Screenshot 2025-02-11 at 2 08 21 PM" src="https://github.com/user-attachments/assets/e4f846c7-821b-4037-86b8-d9c492d0f088" />

---
**Final Model Selected:** **Ridge or Lasso**  
High-Quality Model: Both Ridge and Lasso regression are high-quality models that provide valuable insights into the factors influencing car prices. These models are effective, interpretable, and offer practical value to the business in terms of understanding key drivers of car prices.

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
R² for Ridge and Lasso: Measures how well the model fits the training and test data. A high R² value on both training and test sets indicates a good model fit.

---

### Feature Importance Analysis  
<img width="1124" alt="Screenshot 2025-02-11 at 1 34 09 PM" src="https://github.com/user-attachments/assets/128acdf2-57ae-4e2f-873f-0e8819454120" />

#### Top Factors Influencing Car Prices:  
**1. Odometer (Mileage) - Strongest Predictor**    
   - Lower mileage = Higher price.  
**2. Vehicle Age (Year) - Key Factor**   
   - Newer cars have higher resale value.  
**3. Manufacturer & Model - Price Varies by Brand**   
   - Luxury brands (BMW, Mercedes) retain value better. 
**4. Fuel Gas and Electric - Price Varies by Fuel**   
   - Gas and Electric cars have higher resale value.
     
**Other considerations**
**5. Condition & Fuel Type Impact Pricing** ⚡  
   - Electric & hybrid cars are priced higher.  
**6. Drive Type & Transmission - AWD/4WD Add Value**   
   - More valuable in snowy/off-road regions.

### Model Deployment (Optional)
You can deploy the trained model using Flask or FastAPI. 

## Conclusion
The models evaluated include Linear, Polynomial, Ridge, Lasso regression, and Gradient Boosting. Both Ridge and Lasso regression are high-quality models that provide valuable insights into the factors influencing car prices. These models are effective, interpretable, and offer practical value to the business in terms of understanding key drivers of car prices.
The models provide valuable insights into how various features influence car prices, such as:

Year: Newer cars tend to have higher prices.
Odometer: Higher mileage decreases the price.
Manufacturer: Certain manufacturers tend to have higher price points.
Fuel type: Gas and electric cars have better resale value.

Future Improvements
Explore non-linear models such as Random Forests or Gradient Boosting Machines.
Incorporate additional features like car condition or historical sales data for better accuracy.

As part of our project to predict used car prices, we’ve applied several regression models to understand the key drivers of car prices. The goal of this work is to provide actionable insights that can help used car dealers fine-tune their inventory and optimize pricing strategies. This report outlines the main findings from the modeling process, including which features drive car prices, the quality of our models, and how these insights can be used to adjust the pricing strategy.



## Actionable Insights for Used Car Dealers
Based on the analysis and model findings, here are actionable insights for fine-tuning your inventory and optimizing pricing strategies:

**Pricing for Newer Cars:** Focus on newer vehicles in your inventory, as they tend to have a higher price. Dealers can use the model to predict the price of new cars and ensure they are not underpricing valuable inventory.

**Impact of Mileage:** For cars with high mileage, dealers should consider offering discounts or pricing them more competitively. High-mileage cars tend to have lower prices, and dealers can use this insight to set more accurate pricing and avoid overpricing vehicles with higher usage.

**Brand or Manufacturer Considerations:** Focus on cars from high-demand manufacturers. Certain brands, due to their perceived reliability or brand value, may warrant higher prices. Understanding which manufacturers contribute most to price can help dealers adjust their inventory to meet market demand.

**Optimizing Inventory Based on Model Predictions:** By applying this model to the entire inventory, dealers can forecast the optimal price for each car, identify overpriced or underpriced cars, and make adjustments accordingly. This can help increase sales and reduce unsold inventory.
