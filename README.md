# 🏥 Medical Insurance Cost Prediction

## 📌 Project Overview
This project focuses on predicting **medical insurance charges** for individuals based on demographic and lifestyle factors.  
The aim is to understand the impact of features such as age, smoking habits, BMI, and family size on healthcare costs and to build a machine learning model that can accurately estimate medical expenses.  

---

## 📂 Dataset Description
The dataset contains information about individuals and their respective medical charges.  

| Variable  | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| age       | Age of the individual.                                                     |
| sex       | Gender (male/female).                                                      |
| bmi       | Body Mass Index – weight (kg) / height (m)^2.                              |
| children  | Number of children/dependents covered by insurance.                        |
| smoker    | Whether the individual is a smoker (yes/no).                               |
| region    | Residential region in the US (northeast, northwest, southeast, southwest). |
| charges   | Final medical insurance charges billed to the individual.                  |

---

## 🔎 Analysis Workflow
1. **Data Cleaning & Preprocessing**  
   - Handle categorical encoding (sex, smoker, region).  
   - Check for missing/null values.  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of charges across age, smoker status, and BMI.  
   - Correlation analysis between features and medical charges.  
   - Visualization of key factors impacting costs.  

3. **Feature Engineering**  
   - Encode categorical variables.  
   - Scale/normalize numerical features if required.  

4. **Modeling**  
   - Train regression models to predict `charges`.  
   - Models tested: Linear Regression, Random Forest Regressor, XGBoost.  
   - Evaluate using metrics: MAE, MSE, RMSE, R².  

5. **Residual Analysis**  
   - Compare predicted vs actual charges.  
   - Plot residuals to check model fit.  

6. **Deployment**  
   - Built a **Streamlit App** to allow interactive predictions:  
     User inputs (age, sex, BMI, smoker, children, region) → Predicted Insurance Charges.  

---

## 📊 Tools & Libraries
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn , Plotly
- **Models:** Linear Regression, Random Forest  
- **Visualization:** Plotly 
- **Deployment:** Streamlit  

---

## 🚀 Results
- **Smoker status** and **age** are the most influential factors on medical charges.  
- Model achieved strong predictive performance with R² > 0.80 on test data.  
- Residual analysis confirmed good model fit.  
- Interactive Streamlit app allows real-time insurance charge predictions.  

---

bYE
