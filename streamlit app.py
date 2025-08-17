
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Files ---
model = pickle.load(open(r"E:\Projects\python\Data_Analysis_Machine_Learning\4. Public Health and Safety\Medical Insurance Prediction\linear_model.pkl", "rb"))
X_test = pd.read_pickle(r"E:\Projects\python\Data_Analysis_Machine_Learning\4. Public Health and Safety\Medical Insurance Prediction\X_test.pkl")
y_test = pd.read_pickle(r"E:\Projects\python\Data_Analysis_Machine_Learning\4. Public Health and Safety\Medical Insurance Prediction\y_test.pkl")

# --- Sidebar ---
st.sidebar.title("👤 User Input")
age = st.sidebar.slider("Age", 18, 65, 30)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# --- Main Page ---
st.title("💊 Medical Insurance Cost Prediction")

def preprocess_input():
    data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,         
        "smoker": smoker,   
        "region": region    
    }
    return pd.DataFrame([data])

input_df = preprocess_input()


if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.subheader(f"📊 Predicted Charges: ${pred:,.2f}")

# --- Model Evaluation ---
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)     
rmse = np.sqrt(mse)                             

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("MAE", f"{mae:,.2f}")
col3.metric("RMSE", f"{rmse:,.2f}")
